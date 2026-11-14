import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import gc
from PIL import Image
from transformers import ( Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig)
from qwen_vl_utils import process_vision_info
heightallowed , widthallowed = 784 , 1024 # means 802,816 max pixels
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"loading {model_name}...")
bnb_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4" )
model = Qwen2_5_VLForConditionalGeneration.from_pretrained( model_name,quantization_config=bnb_config,device_map="auto")  # loads the model onto the gpu
processor = AutoProcessor.from_pretrained(model_name,min_pixels=256 * 28 * 28,max_pixels=1024 * 28 * 28) # 256 * 28 * 28 = 200,704 pixels image. 512 * 28 * 28 = 802,816 pixels image .
print("model loded.")

def describe_image(image_path, about: str = "", prompt: str = "Describe this image.", max_new_tokens: int = 500):
    torch.cuda.empty_cache()
    gc.collect()
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    width_exceed = width > widthallowed
    height_exceed = height > heightallowed
    resize = width_exceed or height_exceed
    if resize:
        if height_exceed and (not width_exceed or height / heightallowed >= width / widthallowed):
            height_new = heightallowed
            width_new = int(width / (height / heightallowed))
        else:
            width_new = widthallowed
            height_new = int(height / (width / widthallowed))
        img = img.resize((width_new, height_new))  
    full_prompt = f"Context about this image: {about}\n\n{prompt}" if about else prompt
    try:
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": full_prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        torch.cuda.empty_cache()
        gc.collect()
    return response

if __name__ == "__main__":
    while True :
        image_location = input("Enter the image location for the model to see : ")
        if not  os.path.exists(image_location):
            print("image not found , try to enter the correct location ")
        Image_Prompt = input("Enter the image question for the model to be asked (optional) : ")
        Image_description = input("Enter the image question for the model to be told (optional) : ")
        print(describe_image(image_path=image_location,about=Image_description if Image_description else None , prompt = Image_Prompt if Image_Prompt else None ))