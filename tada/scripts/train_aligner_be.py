import os
# Включаем hf_transfer для максимально быстрой загрузки датасетов
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import torchaudio
import argparse
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor
)
from datasets import load_dataset, Dataset

from tada.modules.aligner import Aligner, AlignerConfig

@dataclass
class DataCollatorCTCWithPadding:
    tokenizer: AutoTokenizer
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Train TADA Aligner for Belarusian")
    parser.add_argument("--subset", action="store_true", help="Use a 5% subset for a quick sanity check")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for downloading and pushing")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Configuration
    base_model_name = "ales/wav2vec2-cv-be" 
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    output_dir = "./tada-aligner-be"
    
    # Optional login to Hugging Face
    if args.hf_token:
        from huggingface_hub import login
        print("Logging into Hugging Face Hub...")
        login(token=args.hf_token)
    
    print("Loading tokenizer and feature extractor...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model_name, token=args.hf_token)

    print("Initializing Aligner and injecting pre-trained Belarusian weights...")
    config = AlignerConfig(
        base_model_name=base_model_name,
        tokenizer_name=tokenizer_name
    )
    aligner = Aligner(config)
    
    # Загружаем предобученные веса, игнорируя размер головы
    pretrained_encoder = AutoModelForCTC.from_pretrained(
        base_model_name, 
        ignore_mismatched_sizes=True, 
        vocab_size=len(tokenizer),
        token=args.hf_token
    )
    
    aligner.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)
    model = aligner.encoder 

    # 2. Data Preparation
    print("Loading datasets/fosters/be-bel-audio-corpus...")
    
    # Чтобы не качать 21 ГБ, если запрошен subset, мы просим у HF только первые 5%
    split_name = "train[:5%]" if args.subset else "train"
    if args.subset:
        print(f"Subset flag detected! Downloading ONLY 5% of the data using split '{split_name}'...")
        
    dataset = load_dataset(
        "fosters/be-bel-audio-corpus", 
        split=split_name, 
        token=args.hf_token,
        # num_proc=4 # Раскомментируйте, если загрузка зависает
    )

    # Сплит на train и eval (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    def prepare_dataset(batch):
        audio = batch["audio"]
        
        # Resample if not 16kHz
        if audio["sampling_rate"] != 16000:
            import torchaudio.functional as F
            waveform = torch.tensor(audio["array"]).float().unsqueeze(0)
            waveform = F.resample(waveform, audio["sampling_rate"], 16000).squeeze(0)
            audio_array = waveform.numpy()
        else:
            audio_array = audio["array"]
            
        batch["input_values"] = feature_extractor(audio_array, sampling_rate=16000).input_values[0]
        batch["labels"] = tokenizer(batch["sentence"], add_special_tokens=False).input_ids
        return batch

    print("Processing and resampling audio (this may take a while)...")
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=4)
    eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names, num_proc=4)
    
    data_collator = DataCollatorCTCWithPadding(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
    )

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8 // args.batch_size, 
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        learning_rate=1e-4, 
        warmup_steps=500,
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="tensorboard",
        # hub_token=args.hf_token # Если захотите потом пушить модель напрямую через Trainer
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 5. Train and Save
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final aligner to {output_dir}...")
    aligner.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done! Model is ready for TADA.")

if __name__ == "__main__":
    main()
