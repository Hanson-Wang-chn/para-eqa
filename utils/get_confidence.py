# utils/get_confidence.py

import logging
from utils.vlm_api import VLM_API
from utils.image_processor import decode_image


def get_confidence(question_desc, kb, prompt_get_confidence, model_api="qwen/qwen2.5-vl-72b-instruct", use_openrouter=False):
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
    
    # 构造提示
    prompt = prompt_get_confidence.format(question_desc)
    
    # 调用VLM
    response = vlm.request_with_retry(image=None, prompt=prompt, kb=kb)[0]
    
    # 选项和数值的对照表
    choices_mapping = {
        "A": 0.1,
        "B": 0.3,
        "C": 0.5,
        "D": 0.7,
        "E": 0.9
    }
    
    # 解析响应获取置信度
    try:
        confidence = choices_mapping.get(response.strip().upper(), -1.0)
        if confidence == -1.0:
            logging.error(f"无法从VLM响应中解析置信度: {response}")
            return 0.0
        
        # 确保置信度在[0,1]范围内
        confidence = max(0.0, min(1.0, confidence))
        return confidence
    
    except ValueError:
        logging.error(f"无法从VLM响应中解析置信度: {response}")
        return 0.0


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
