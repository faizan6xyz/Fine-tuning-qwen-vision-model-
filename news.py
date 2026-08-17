import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import gc
import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from PIL import Image
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from qwen_vl_utils import process_vision_info
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_FILE = "Data/train.jsonl"
IMAGE_ROOT = "Data/images/images"
OUTPUT_DIR = "out/qwen25vl-3b-lora"
MAX_PIXELS = 256 * 256
IGNORE_INDEX = -100
from pathlib import Path
class VLSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str):
        self.image_root = Path(image_root)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
    def __len__(self) -> int:
        return len(self.records)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image_filename = rec["image_name"].strip().replace("\\", "/")  # normalize any stray backslashes from the JSONL itself
        image_path = self.image_root / image_filename
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found for record {idx}: '{image_path}' "
                f"(image_root='{self.image_root}', image_name='{rec['image_name']}')"
            )
        img = Image.open(image_path).convert("RGB").resize((448, 448))
        return {"image": img, "prompt": rec["prompt"], "response": rec["label"]}
@dataclass
class VLDataCollator:
    processor: Any
    max_pixels: int = MAX_PIXELS
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = []
        prompt_texts = []
        all_images = []
        for ex in batch:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": ex["image"], "max_pixels": self.max_pixels},
                    {"type": "text", "text": ex["prompt"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ex["response"]},
                ]},
            ]
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.processor.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            full_texts.append(full_text)
            prompt_texts.append(prompt_text)
            all_images.extend(image_inputs)
        model_inputs = self.processor(
            text=full_texts,
            images=all_images,
            padding=True,
            return_tensors="pt",
        )
        labels = model_inputs["input_ids"].clone()
        for i, prompt_text in enumerate(prompt_texts):
            prompt_len = len(self.processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            labels[i, :prompt_len] = IGNORE_INDEX
        pad_id = self.processor.tokenizer.pad_token_id
        labels[model_inputs["input_ids"] == pad_id] = IGNORE_INDEX
        model_inputs["labels"] = labels
        return model_inputs
if __name__ == "__main__":
    print(f"Loading {model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print("Model loaded successfully.")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"], 
    )
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    torch.cuda.empty_cache()
    gc.collect()
    train_dataset = VLSFTDataset(TRAIN_FILE, IMAGE_ROOT)
    collator = VLDataCollator(processor=processor, max_pixels=MAX_PIXELS)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")
    torch.cuda.empty_cache()
    gc.collect()