import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
MODEL_ID = "Qwen/Qwen2.5-VL-8B-Instruct"
IGNORE_INDEX = -100
class VLSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str):
        self.image_root = image_root
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
    def __len__(self) -> int:
        return len(self.records)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image_path = os.path.join(self.image_root, rec["image"])
        human_turns = [c["value"] for c in rec["conversations"] if c["from"] == "human"]
        gpt_turns = [c["value"] for c in rec["conversations"] if c["from"] == "gpt"]
        assert human_turns and gpt_turns, f"Malformed record at index {idx}: {rec}"
        instruction = human_turns[0]
        response = gpt_turns[-1]
        return {
            "image_path": image_path,
            "instruction": instruction,
            "response": response,
        }
@dataclass
class VLDataCollator:
    processor: Any
    max_pixels: int = 512 * 512
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        image_inputs_all = []
        for ex in batch:
            image = Image.open(ex["image_path"]).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "max_pixels": self.max_pixels},
                        {"type": "text", "text": ex["instruction"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["response"]}],
                },
            ]
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_only = self.processor.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True
            )
            texts.append((full_text, prompt_only))
            image_inputs, _ = process_vision_info(messages)
            image_inputs_all.append(image_inputs)
        full_texts = [t[0] for t in texts]
        prompt_texts = [t[1] for t in texts]
        model_inputs = self.processor(
            text=full_texts,
            images=[img for imgs in image_inputs_all for img in imgs],
            padding=True,
            return_tensors="pt",
        )
        labels = model_inputs["input_ids"].clone()
        for i, prompt_text in enumerate(prompt_texts):
            prompt_ids = self.processor.tokenizer(
                prompt_text, add_special_tokens=False
            )["input_ids"]
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = IGNORE_INDEX
        pad_id = self.processor.tokenizer.pad_token_id
        labels[model_inputs["input_ids"] == pad_id] = IGNORE_INDEX
        model_inputs["labels"] = labels
        return model_inputs
def load_model_and_processor(max_pixels: int):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = QwenVLModel.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 256,
        max_pixels=max_pixels,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--image_root", type=str, default="data/images")
    parser.add_argument("--output_dir", type=str, default="out/qwen25vl-8b-lora")
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--merged_dir", type=str, default="out/merged")
    args = parser.parse_args()
    if args.merge_only:
        merge_lora(args.output_dir, args.merged_dir)
        return
    model, processor = load_model_and_processor(args.max_pixels)
    train_dataset = VLSFTDataset(args.train_file, args.image_root)
    collator = VLDataCollator(processor=processor, max_pixels=args.max_pixels)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,  
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")
    print("Run again with --merge_only to bake the adapter into a standalone model.")
def merge_lora(adapter_dir: str, merged_dir: str):
    print("Loading base model in fp16 (no quantization) for a clean merge...")
    base_model = QwenVLModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    processor = AutoProcessor.from_pretrained(adapter_dir)
    processor.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")
if __name__ == "__main__":
    main()