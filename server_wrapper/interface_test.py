import requests
import base64
import io
from PIL import Image

# 读取图像并转为 base64 编码
img = Image.open("vlm/demo/demo.jpeg")
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format="PNG")  # 可以指定图片格式
img_byte_arr.seek(0)

question = "Is the red apple placed on top of the cardboard box?"
data = {
    "text": f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view? Answer with Yes or No.",
}
files = {'image': img_byte_arr}

response = requests.post("http://127.0.0.1:5000/get_response", files=files, data=data)

# 打印返回的结果
print(response.status_code)
print(response.text)
