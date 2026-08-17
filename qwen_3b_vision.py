import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import gc
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"Loading {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    model_name,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28
)
print("Model loaded successfully.")
def describe_image(image_path, prompt: str = "Describe this image.", max_new_tokens: int = 300):
    torch.cuda.empty_cache()
    gc.collect()
    img = Image.open(image_path).convert("RGB")
    img = img.resize((448, 448))
    folder_path = os.path.dirname(image_path)
    resized_path = os.path.join(folder_path, "temp_resized.jpg")
    img.save(resized_path)
    try:
        messages = {
        "messages": [
            {"role": "user", "content": [
            {"type": "image", "image": "path/to/img.jpg"},
            {"type": "text", "text": "What is shown in this image?"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ]}
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
    finally:
        if os.path.exists(resized_path):
            os.remove(resized_path)
    torch.cuda.empty_cache()
    gc.collect()
    return response
if __name__ == "__main__":
    result, path = describe_image("h.png")
    print(result)