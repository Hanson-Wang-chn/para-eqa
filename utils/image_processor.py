from PIL import Image
import base64
import logging
import io


def decode_image(base64_string):
    """将Base64编码的图像解码为PIL Image对象"""
    if not base64_string:
        return None
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logging.error(f"图像解码错误: {e}")
        return None


def encode_image(image):
    """将PIL Image对象编码为Base64字符串"""
    if image is None:
        return None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"图像编码错误: {e}")
        return None
