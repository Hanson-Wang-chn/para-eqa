# utils/get_confidence.py

import logging
from utils.vlm_api import VLM_API
from utils.image_processor import decode_image


def get_confidence(question_desc, kb, prompt_get_confidence, model_api="gpt-4.1", use_openrouter=False):
    """
    调用大模型计算能否回答问题的置信度
    
    Args:
        question_desc (str): 问题描述
        kb (list): 从记忆中检索到的项目列表
        prompt_get_confidence (str): 提示模板
        model_api (str): 使用的OpenAI模型
        
    Returns:
        float: 置信度，范围[0,1]
    """    
    # 实例化VLM
    vlm = VLM_API(model_name=model_api, use_openrouter=use_openrouter)
    
    # TODO:
    prompt = concatinate(question_desc, prompt_get_confidence)
    
    # 调用VLM
    response = vlm.request_with_retry(image=None, prompt=prompt, kb=kb)[0]
    
    # 解析响应获取置信度
    try:
        confidence = float(response.strip())
        # 确保置信度在[0,1]范围内
        confidence = max(0.0, min(1.0, confidence))
        return confidence
    except ValueError:
        logging.error(f"无法从VLM响应中解析置信度: {response}")
        return 0.0  # 默认为0，表示没有信心


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
