# utils/get_confidence.py

import logging
from utils.vlm_api import VLM_API
from utils.image_processor import decode_image


def get_confidence(question_desc, image, kb, prompt_get_confidence, model_name="qwen/qwen2.5-vl-72b-instruct", server="openrouter", base_url=None, api_key=None):
    # FIXME:
    # return 1.0
    """
    调用大模型计算能否回答问题的置信度
    
    Args:
        question_desc (str): 问题描述
        image (str): 图片数据，base64编码的字符串，可以为空
        kb (list): 从记忆中检索到的项目列表，可以为空
        prompt_get_confidence (str): 提示模板
        model_name (str): 使用的模型名称
        server (str): 服务器类型
        base_url (str): 基础URL
        api_key (str): API密钥
        
    Returns:
        float: 置信度，范围[0,1]
    """    
    # 实例化VLM
    vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
    
    # 构造提示
    prompt = prompt_get_confidence.format(question_desc)
    
    # 选项和数值的对照表
    choices_mapping = {
        "A": 0.1,
        "B": 0.3,
        "C": 0.5,
        "D": 0.7,
        "E": 0.9
    }
    
    # 最多重试5次
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # 调用VLM
            response = vlm.request_with_retry(image=image, prompt=prompt, kb=kb)[0]
            
            # 解析响应获取置信度
            confidence = choices_mapping.get(response.strip().upper(), -1.0)
            
            if confidence != -1.0:
                # 成功解析，确保置信度在[0,1]范围内并返回
                confidence = max(0.0, min(1.0, confidence))
                if attempt > 0:
                    logging.info(f"第{attempt + 1}次尝试成功解析置信度: {confidence}")
                return confidence
            
            else:
                logging.warning(f"第{attempt + 1}次尝试无法解析置信度，VLM响应: {response}")
        
        except ValueError as e:
            logging.warning(f"第{attempt + 1}次尝试解析置信度时发生错误: {e}, VLM响应: {response}")
        
        except Exception as e:
            logging.error(f"第{attempt + 1}次尝试调用VLM时发生错误: {e}")
    
    # 所有重试都失败，返回默认值
    logging.error(f"经过{max_retries}次尝试后仍无法从VLM响应中解析置信度，返回默认值0.0")
    return 0.0


def get_tryout_confidence(question_desc, image, kb, prompt_get_tryout_answer, prompt_get_tryout_confidence, model_name="qwen/qwen2.5-vl-72b-instruct", server="openrouter", base_url=None, api_key=None):
    """
    先让大模型尝试回答该问题，然后根据猜测的回答判断置信度
    
    Args:
        question_desc (str): 问题描述
        image (str): 图片数据，base64编码的字符串，可以为空
        kb (list): 从记忆中检索到的项目列表，可以为空
        prompt_get_tryout_answer (str): 提示模板
        prompt_get_tryout_confidence (str): 提示模板
        model_name (str): 使用的模型名称
        server (str): 服务器类型
        base_url (str): 基础URL
        api_key (str): API密钥
        
    Returns:
        float: 置信度，范围[0,1]
    """
    # 实例化VLM
    vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
    
    # 构造第一个提示 - 获取猜测答案
    prompt_answer = prompt_get_tryout_answer.format(question_desc)
    
    # 选项和数值的对照表
    choices_mapping = {
        "A": 0.1,
        "B": 0.3,
        "C": 0.5,
        "D": 0.7,
        "E": 0.9
    }
    
    # 最多重试5次
    max_retries = 5
    
    # 第一步：获取猜测答案
    answer_tryout = None
    
    for attempt in range(max_retries):
        try:
            # 调用VLM获取猜测答案
            response = vlm.request_with_retry(image=image, prompt=prompt_answer, kb=kb)[0]
            
            if response is None:
                logging.warning(f"第{attempt + 1}次尝试获取猜测答案时，VLM响应为None")
                continue
                
            # 解析响应获取猜测答案
            answer_tryout = response.strip()
            
            if answer_tryout:
                logging.info(f"成功获取猜测答案: {answer_tryout}")
                if attempt > 0:
                    logging.info(f"第{attempt + 1}次尝试成功获取猜测答案")
                break
            
            else:
                logging.warning(f"第{attempt + 1}次尝试无法获取猜测答案，VLM响应为空字符串")
        
        except Exception as e:
            logging.error(f"第{attempt + 1}次尝试调用VLM获取猜测答案时发生错误: {e}")
    
    # 如果无法获取猜测答案，返回默认置信度
    if not answer_tryout:
        logging.error(f"经过{max_retries}次尝试后仍无法获取猜测答案，返回默认值0.0")
        return 0.0
    
    # 第二步：根据猜测答案获取置信度
    # 构造第二个提示 - 获取置信度
    prompt_confidence = prompt_get_tryout_confidence.format(question_desc=question_desc, tryout_answer=answer_tryout)
    
    for attempt in range(max_retries):
        try:
            # 调用VLM获取置信度
            response = vlm.request_with_retry(image=image, prompt=prompt_confidence, kb=kb)[0]
            
            if response is None:
                logging.warning(f"第{attempt + 1}次尝试获取置信度时，VLM响应为None")
                continue
                
            # 解析响应获取置信度
            confidence = choices_mapping.get(response.strip().upper(), -1.0)
            
            if confidence != -1.0:
                # 成功解析，确保置信度在[0,1]范围内并返回
                confidence = max(0.0, min(1.0, confidence))
                if attempt > 0:
                    logging.info(f"第{attempt + 1}次尝试成功解析置信度: {confidence}")
                return confidence
            
            else:
                logging.warning(f"第{attempt + 1}次尝试无法解析置信度，VLM响应: {response}")
        
        except ValueError as e:
            logging.warning(f"第{attempt + 1}次尝试解析置信度时发生错误: {e}, VLM响应: {response}")
        
        except Exception as e:
            logging.error(f"第{attempt + 1}次尝试调用VLM获取置信度时发生错误: {e}")
    
    # 所有重试都失败，返回默认值
    logging.error(f"经过{max_retries}次尝试后仍无法从VLM响应中解析置信度，返回默认值0.0")
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
