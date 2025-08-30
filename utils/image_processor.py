# utils/image_processor.py

from PIL import Image
import base64
import logging
import io


def decode_image(base64_string):
    """Decode Base64 encoded image to PIL Image object"""
    if not base64_string:
        return None
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logging.error(f"Image decoding error: {e}")
        return None


def encode_image(image):
    """Encode PIL Image object to Base64 string"""
    if image is None:
        return None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Image encoding error: {e}")
        return None
