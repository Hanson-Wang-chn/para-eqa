from flask import Flask, request, jsonify
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import time
from PIL import Image
import io
import base64
import os 
from qwen_vl_utils import process_vision_info
torch.manual_seed(1234)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load tokenizer and model
model_name = "/home/zml/model/Qwen2-VL-72B-Instruct-GPTQ-Int4"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_name)

@app.route('/process', methods=['POST'])
def process():
    try:
        text = request.form['text']
        if 'image' in request.files:
            image_file = request.files['image']
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if 'image' in request.files:
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to("cuda")

        print(inputs)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        return jsonify({"response": output_text})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)