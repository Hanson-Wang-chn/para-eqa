# utils/vlm_api.py

import http
import base64
import http.client
import json
from PIL import Image
import logging
from io import BytesIO
import time
import os
import openai
import httpx


class VLM_API:
    def __init__(self, model_name="openai/gpt-4o", server="openrouter", base_url=None, api_key=None):
        self.model_name = model_name
        
        # 初始化OpenAI客户端，显式设置代理
        proxies = {
            "http://": "socks5://127.0.0.1:7897",
            "https://": "socks5://127.0.0.1:7897"
        } if os.environ.get('ALL_PROXY') else None
        
        self.server = server
        self.base_url = base_url
        self.api_key = None
        self.client = None
        
        if self.server == "openai":
            try:
                self.api_key = os.environ.get('OPENAI_API_KEY') if not api_key else api_key
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is not set.")
            except KeyError:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url = "https://api.openai.com/v1" if not self.base_url else self.base_url,
                http_client=httpx.Client(proxies=proxies) if proxies else None
            )
        
        elif self.server == "openrouter":
            try:
                self.api_key = os.environ.get('OPENROUTER_API_KEY') if not api_key else api_key
                if not self.api_key:
                    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
            except KeyError:
                raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
            
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1" if not self.base_url else self.base_url,
                http_client=httpx.Client(proxies=proxies) if proxies else None
            )
            
        elif self.server == "dashscope":
            try:
                self.api_key = os.environ.get('DASHSCOPE_API_KEY') if not api_key else api_key
                if not self.api_key:
                    raise ValueError("DASHSCOPE_API_KEY environment variable is not set.")
            except KeyError:
                raise ValueError("DASHSCOPE_API_KEY environment variable is not set.")
            
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" if not self.base_url else self.base_url,
                http_client=httpx.Client(proxies=proxies) if proxies else None
            )
        
        elif self.server == "ollama":
            self.client = openai.OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1" if not self.base_url else self.base_url,
                http_client=httpx.Client(proxies=proxies) if proxies else None
            )
        
        else:
            # OpenAI compatible only
            pass


    def convert_file_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def convert_PIL_to_base64(self, image):
        if image is None:
            return None
        
        # 检查图片尺寸，若长或宽小于32，则缩放（适配Ollama）
        min_size = 32
        w, h = image.size
        if w < min_size or h < min_size:
            new_w = max(w, min_size)
            new_h = max(h, min_size)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        with BytesIO() as output:
            image.save(output, format="PNG")
            base64_image = base64.b64encode(output.getvalue()).decode('utf-8')
        return base64_image


    def convert_base64(self, image):
        if image is None:
            return None
        elif isinstance(image, str):
            base64_image = self.convert_file_to_base64(image)
        elif isinstance(image, Image.Image):
            base64_image = self.convert_PIL_to_base64(image)
        else:
            return None
        return base64_image
    
    
    def prepare_data(self, image, prompt, kb):
        # System message to define the role of KB
        system_message = {
            "role": "system",
            "content": "You are an AI assistant that answers questions based on provided information. You will receive a user question along with optional images and reference materials from a knowledge base (KB). When KB materials are provided, use them as authoritative sources to inform your response. If the KB contains relevant information, prioritize it in your answer."
        }
        
        # Start building user content with the prompt
        user_content = [
            {
                "type": "text",
                "text": "\nUser question: \n" + prompt
            }
        ]
        
        # Handle image parameter (can be single image or list of images)
        if image is not None:
            if isinstance(image, list):
                # Image list
                for img in image:
                    if img is not None:  # Skip None values in list
                        base64_image = self.convert_base64(img)
                        if base64_image:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            })
            else:
                # Single image
                base64_image = self.convert_base64(image)
                if base64_image:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
        
        # Handle knowledge base (KB) - add to user content if not empty
        if kb and len(kb) > 0:
            user_content.append({
                "type": "text",
                "text": "\n\nReference materials from knowledge base:"
            })
            
            for i, item in enumerate(kb, 1):
                # Add KB text description
                if item.get('text', None):
                    user_content.append({
                        "type": "text",
                        "text": f"\nReference {i}: {item['text']}"
                    })
                
                # Add KB image if present
                if item.get('image', None):
                    # Assume item['image'] is already base64 encoded
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{item['image']}"
                        }
                    })
        
        # Build user message
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        # Return both system and user messages
        return [system_message, user_message]


    def has_valid_images(self, image):
        """检查是否有有效的图片"""
        if image is None:
            return False
        elif isinstance(image, list):
            # 检查列表中是否有非None的图片
            return any(img is not None for img in image)
        else:
            # 单张图片
            return True


    def request_with_retry(self, image, prompt, kb=[], retries=10):
        def exponential_backoff(attempt):
            return min(2 ** attempt, 60)
        
        if not self.has_valid_images(image) and not kb:
            # 没有有效图片，使用纯文本API
            for attempt in range(retries):
                try:
                    return self.requests_api_only_text(prompt)
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = exponential_backoff(attempt)
                        logging.error(f"Request failed, retrying in {wait_time} seconds... {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
        
        else:
            for attempt in range(retries):
                try:
                    return self.requests_api(image, prompt, kb)
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = exponential_backoff(attempt)
                        logging.error(f"Request failed, retrying in {wait_time} seconds... {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
    
    
    def requests_api(self, image, prompt, kb):
        messages = self.prepare_data(image, prompt, kb)
        
        # Send request using OpenAI official client
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.0
        )
        
        # Extract response content - keep the same return format as original
        return [response.choices[0].message.content]
    
    
    def requests_api_only_text(self, prompt):        
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # 使用OpenAI官方客户端发送请求
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=1000,
            temperature=0.0
        )
        
        # 提取响应内容
        return [response.choices[0].message.content]
    
    
if __name__ == "__main__":
    # 测试API调用是否正常
    model_name = "qwen2.5-vl-7b-instruct"
    server = "dashscope"
    base_url = None
    api_key = None
    vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
    response = vlm.request_with_retry(image=None, prompt="Hello!")[0]
    print(response)


"""
kb格式：
[
    {
        "id": str,
        "text": str,
        "image": str,  # base64编码的图片字符串
    },
    ...
]
"""
