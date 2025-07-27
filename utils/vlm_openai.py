import json
import http
import base64
import http.client
import json
# import PIL
from PIL import Image
import logging
from io import BytesIO
import time
import os
import openai
import httpx

class VLM_OpenAI:
    def __init__(self, model_name="gpt-4.1"):
        self.api_key = os.environ.get('OPENAI_API_KEY', 'sk-your-default-key')
        self.model_name = model_name
        
        # 初始化OpenAI客户端，显式设置代理
        proxies = {
            "http://": "socks5://127.0.0.1:7897",
            "https://": "socks5://127.0.0.1:7897"
        } if os.environ.get('ALL_PROXY') else None
        self.client = openai.OpenAI(
            api_key=self.api_key,
            http_client=httpx.Client(proxies=proxies) if proxies else None
        )


    def convert_file_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def convert_PIL_to_base64(self, image):
        if image is None:
            return None
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
        return base64_image
    
    
    def prepare_data(self, image, prompt):    
        base64_image = self.convert_base64(image)
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        return message


    def request_with_retry(self, image, prompt, retries=10):
        def exponential_backoff(attempt):
            return min(2 ** attempt, 60)
        
        if image is None:
            for attempt in range(retries):
                try:
                    return self.requests_api_only_text(prompt)
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = exponential_backoff(attempt)
                        logging.log(logging.ERROR, f"Request failed, retrying in {wait_time} seconds... {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
        
        else:
            for attempt in range(retries):
                try:
                    return self.requests_api(image, prompt)
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = exponential_backoff(attempt)
                        logging.log(logging.ERROR, f"Request failed, retrying in {wait_time} seconds... {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
    
    
    def requests_api(self, image, prompt):
        message = self.prepare_data(image, prompt)
        
        # 使用OpenAI官方客户端发送请求
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=1000
        )
        
        # 提取响应内容
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
            max_tokens=1000
        )
        
        # 提取响应内容
        return [response.choices[0].message.content]
    
    
if __name__ == "__main__":
    # 简单测试API是否可用
    vlm = VLM_OpenAI()
    try:
        result = vlm.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10
        )
        print("API调用成功，返回内容：", result.choices[0].message.content)
    except Exception as e:
        print("API调用失败：", e)
        