# services/parser_service.py

import os
import uuid
import json
import time
import logging
import re

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.vlm_api import VLM_API


def parse_question(desc, model_name="openai/gpt-oss-120b", prompt_parser=None, server="openrouter", base_url=None, api_key=None):
    """
    解析问题描述，使用大模型提取urgency和scope_type
    
    Args:
        desc (str): 问题的自然语言描述
        model_api (str): 使用的OpenAI模型名称
        
    Returns:
        tuple: (urgency, scope_type)
            - urgency (float): 紧急程度，范围[0,1]
            - scope_type (str): 范围类型，"local"或"global"
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        # 调用大模型获取回答
        vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
        prompt = prompt_parser.replace("{original_question}", desc)
        response = vlm.request_with_retry(image=None, prompt=prompt)[0]
        
        # 提取JSON部分
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = response.strip()
        
        # 解析JSON
        try:
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError:
            logging.info(f"第{attempt + 1}次尝试失败，返回的不是有效JSON: {json_str}")
            valid = False
            continue
        
        # 提取数据
        urgency = parsed_data.get('urgency')
        scope_type = parsed_data.get('scope_type')
        
        # 验证取值范围
        valid = True
        
        # 检查urgency是否为[0,1]范围内的浮点数
        if not isinstance(urgency, (int, float)) or urgency < 0 or urgency > 1:
            logging.info(f"Warning: urgency值不符合规则 (应为[0,1]范围内的数值): {urgency}")
            valid = False
        
        # 检查scope_type是否为["local", "global"]中的字符串
        if scope_type not in ["local", "global"]:
            logging.info(f"Warning: scope_type值不符合规则 (应为'local'或'global'): {scope_type}")
            valid = False
        
        # 如果验证通过，返回结果
        if valid:
            return float(urgency), str(scope_type)
        
        # 如果未通过验证，显示重试信息
        logging.info(f"第{attempt + 1}次尝试失败，重新调用大模型...")
    
    # 三次尝试都失败后报错
    raise ValueError(f"经过{max_retries}次尝试，仍无法获得符合规则的解析结果")


def run(config: dict):
    """
    Parser Service 的主运行函数。
    由主入口脚本 (run_system.py) 作为独立进程启动。
    """
    # Set up logging
    parent_dir = config.get("output_parent_dir", "logs")
    logs_dir = os.path.join(parent_dir, "parser_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "parser.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # Set up VLM
    prompt_parser = config.get("prompt", {}).get("parser", None)
    
    config_vlm = config.get("vlm", {}).get("parser", {})
    model_name = config_vlm.get("model", "qwen/qwen2.5-vl-72b-instruct")
    server = config_vlm.get("server", "openrouter")
    base_url = config_vlm.get("base_url", None)
    api_key = config_vlm.get("api_key", None)
    
    # Redis Initialization
    redis_conn = get_redis_connection(config)
    stream_name = STREAMS["generator_to_parser"]
    group_name = "parser_group"

    # 尝试创建消费者组，如果已存在则会报错，但可以安全地忽略
    try:
        redis_conn.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PAR) Parser group '{group_name}' already exists. Continuing...")
        pass

    logging.info(f"[{os.getpid()}](PAR) Parser service started. Waiting for new questions...")
    
    while True:
        try:
            # 阻塞式地从"新问题"流中读取一条消息
            messages = redis_conn.xreadgroup(group_name, "parser_worker", {stream_name: '>'}, count=1, block=None)
            
            if not messages:
                time.sleep(0.01)
                continue

            for stream, message_list in messages:
                for message_id, data in message_list:
                    logging.info(f"[{os.getpid()}](PAR) Parsing message {message_id}...")
                    question_data = json.loads(data['data'])
                    desc = question_data.get("description")
                    q_id = question_data.get("id", str(uuid.uuid4()))
                    
                    if not desc:
                        logging.info(f"[{os.getpid()}](PAR) WARN: Received message without a description. Skipping.")
                        redis_conn.xack(stream_name, group_name, message_id)
                        continue

                    # 1. 创建完整的 Question 元数据
                    urgency, scope_type = parse_question(desc, model_name=model_name, prompt_parser=prompt_parser, server=server, base_url=base_url, api_key=api_key)
                    metadata = {
                        "id": q_id,
                        "description": desc,
                        "urgency": urgency,
                        "scope_type": scope_type,
                        "status": "pending",  # 所有问题的初始状态
                        "cost_estimate": -1.0, # 待Updater更新
                        "reward_estimate": -1.0, # 待Updater更新
                        "dependency": [],
                        "answer": "",
                        "max_steps": 0,
                        "used_steps": 0,
                        "time": {
                            "request": question_data.get("time", {}).get("request", 0)
                        }
                    }
                    logging.info(f"[{os.getpid()}](PAR) Question metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
                    
                    # 2. 将元数据存入 Redis Hash
                    # redis_conn.hset(f"{KEY_PREFIXES['question']}{q_id}", mapping=metadata)
                    
                    # 3. 通知 Finishing Module 服务，有新的问题需要处理
                    redis_conn.xadd(STREAMS["parser_to_finishing"], {"data": json.dumps(metadata)})
                    
                    # 4. 更新统计信息
                    redis_conn.hincrby(STATS_KEYS["parser"], "total", 1)
                    
                    logging.info(f"[{os.getpid()}](PAR) Successfully parsed question {q_id}: '{desc[:40]}...'")

                    # 5. 确认消息处理完毕，从pending列表中移除
                    redis_conn.xack(stream_name, group_name, message_id)

        except Exception as e:
            logging.info(f"[{os.getpid()}](PAR) An error occurred in Parser service: {e}")
            time.sleep(5) # 发生错误时等待一段时间再重试
