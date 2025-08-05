# services/answering_service.py

import os
import json
import time
import logging

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.vlm_api import VLM_API
from utils.image_processor import decode_image


def get_vlm_answer(question, kb, prompt_get_answer, model_api="gpt-4.1", use_openrouter=False):
    """
    根据问题和记忆数据，使用VLM生成答案
    
    Args:
        question (dict): 问题对象，包含描述等信息
        memory_data (list): 从记忆中检索到的数据
        model_api (str): 使用的OpenAI模型名称
        
    Returns:
        str: 生成的答案
    """
    # 构建提示词
    question_desc = question.get('description', '')
    prompt = prompt_get_answer.format(question_desc)
    
    # 实例化VLM并请求回答
    vlm = VLM_API(model_name=model_api, use_openrouter=use_openrouter)
    response = vlm.request_with_retry(image=None, prompt=prompt, kb=kb)[0]
    
    return response.strip()


def run(config: dict):
    """
    Answering Service 的主运行函数。
    负责接收问题，生成答案，并将答案发送到Question Pool。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "results")
    logs_dir = os.path.join(parent_dir, "answering_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "answering.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # 读取配置
    answering_config = config.get("answering", {})
    result_path = answering_config.get("result_path", "results/answers.json")
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # 初始化结果文件，如果不存在
    if not os.path.exists(result_path):
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    # VLM配置
    model_api = config.get("vlm", {}).get("model_api", "gpt-4.1")
    use_openrouter = config.get("vlm", {}).get("use_openrouter", False)
    prompt_get_answer = config.get("prompt", {}).get("answering", {}).get("get_answer", "")
    
    # Redis初始化
    redis_conn = get_redis_connection(config)
    
    # 流定义
    to_answering_stream = STREAMS["to_answering"]  # 从Finishing接收问题
    answering_to_pool_stream = STREAMS["answering_to_pool"]  # 向Question Pool发送答案
    
    # 创建消费者组
    group_name = "answering_group"
    try:
        redis_conn.xgroup_create(to_answering_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Answering group already exists: {e}")
    
    logging.info(f"[{os.getpid()}] Answering service started. Waiting for questions...")
    
    # 初始化统计计数器
    answered_count = 0
    
    while True:
        try:
            # 从to_answering流中读取消息
            messages = redis_conn.xreadgroup(
                group_name, "answering_worker", 
                {to_answering_stream: '>'}, 
                count=1, block=None
            )
            
            if not messages:
                time.sleep(0.01)
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    try:
                        # 解析请求数据
                        request_data = json.loads(data.get('data', '{}'))
                        question = request_data.get('question', {})
                        memory_data = request_data.get('memory_data', [])
                        
                        question_id = question.get('id')
                        question_desc = question.get('description', '')
                        
                        logging.info(f"[{os.getpid()}] 收到问题: {question_id} - '{question_desc[:40]}...'")
                        
                        # 获取答案
                        answer = get_vlm_answer(question, memory_data, prompt_get_answer, model_api, use_openrouter)
                        logging.info(f"[{os.getpid()}] 已生成问题 {question_id} 的答案")
                        
                        # 更新问题元数据
                        question['answer'] = answer
                        
                        # 将答案保存到文件
                        try:
                            # 读取现有的答案
                            existing_answers = []
                            try:
                                with open(result_path, 'r', encoding='utf-8') as f:
                                    existing_answers = json.load(f)
                            except (json.JSONDecodeError, FileNotFoundError):
                                logging.warning(f"[{os.getpid()}] 无法读取现有答案文件，将创建新文件")
                            
                            # 检查是否已存在相同ID的问题
                            for i, existing_answer in enumerate(existing_answers):
                                if existing_answer.get('id') == question_id:
                                    # 更新现有答案
                                    existing_answers[i] = question
                                    break
                            else:
                                # 添加新答案
                                existing_answers.append(question)
                            
                            # 写入更新后的答案
                            with open(result_path, 'w', encoding='utf-8') as f:
                                json.dump(existing_answers, f, ensure_ascii=False, indent=2)
                                
                            logging.info(f"[{os.getpid()}] 问题 {question_id} 的答案已保存到文件")
                        except Exception as e:
                            logging.error(f"[{os.getpid()}] 保存答案到文件时出错: {e}")
                        
                        # 向Question Pool发送完成的问题
                        redis_conn.xadd(answering_to_pool_stream, {"data": json.dumps(question)})
                        logging.info(f"[{os.getpid()}] 问题 {question_id} 的答案已发送到Question Pool")
                        
                        # 更新统计信息
                        answered_count += 1
                        redis_conn.hset(STATS_KEYS["answering"], "answered", answered_count)
                        
                    except Exception as e:
                        logging.error(f"[{os.getpid()}] 处理问题时出错: {e}")
                    
                    # 确认消息处理完毕
                    redis_conn.xack(to_answering_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"[{os.getpid()}] Answering service发生错误: {e}")
            time.sleep(5)  # 发生错误时等待一段时间再重试
