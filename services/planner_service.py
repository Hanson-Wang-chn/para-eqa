# services/planner_service.py

import os
import json
import time
import logging
import uuid

from common.redis_client import get_redis_connection, STREAMS
from utils.para_eqa import ParaEQA


def select_question(redis_conn):
    """
    向Question Pool请求一个优先级最高的问题
    
    Args:
        redis_conn: Redis连接对象
        
    Returns:
        dict: 问题对象，如果没有可用问题则返回None
    """
    # 创建消费者组（如果不存在）
    pool_responses_stream = STREAMS["pool_responses"]
    
    try:
        redis_conn.xgroup_create(pool_responses_stream, "planner_group", id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PLA) Planner response group already exists: {e}")
        pass
    
    while True:
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 创建请求
        request = {
            "request_id": request_id,
            "sender": "planner",
            "type": "select_question",
            "data": {}
        }
        
        # 发送请求
        redis_conn.xadd(STREAMS["pool_requests"], {"data": json.dumps(request)})
        
        # 等待响应
        wait_start_time = time.time()
        max_wait_time = 3  # 单次响应最多等待3秒，若超时，说明planner阻塞（正在处理其他请求）
        
        while time.time() - wait_start_time < max_wait_time:
            responses = redis_conn.xreadgroup(
                "planner_group", "planner_worker", 
                {pool_responses_stream: '>'}, 
                count=20, block=100
            )
            
            if not responses:
                continue
                
            for _, msg_list in responses:
                for msg_id, data in msg_list:
                    try:
                        response = json.loads(data.get('data', '{}'))
                        response_request_id = response.get('request_id')
                        
                        # 确认消息
                        redis_conn.xack(pool_responses_stream, "planner_group", msg_id)
                        
                        # 检查是否是我们请求的响应
                        if response_request_id == request_id and response.get('type') == 'question_selected':
                            if response.get('status') == "success" and response.get('data'):
                                logging.info(f"[{os.getpid()}](PLA) Received question {response['data']['id']} from Question Pool")
                                return response['data']
                            else:
                                return None
                    except Exception as e:
                        logging.error(f"[{os.getpid()}](PLA) Error processing response: {e}")
        
        time.sleep(0.1)


def run(config: dict):
    """
    Planner Service 的主运行函数。
    负责规划探索路径并与环境交互。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "results")
    logs_dir = os.path.join(parent_dir, "planner_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "planner.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # 连接 Redis
    redis_conn = get_redis_connection(config)
    
    logging.info(f"[{os.getpid()}](PLA) Planner service started.")
    
    # 实例化 ParaEQA
    para_eqa = ParaEQA(config)
    
    # 主循环
    while True:
        try:
            question = select_question(redis_conn)
            if question is None:
                logging.info(f"[{os.getpid()}](PLA) No available questions, waiting for new questions...")
                time.sleep(1)
                continue
            para_eqa.run(question, question["id"])
                
        except Exception as e:
            logging.exception(f"[{os.getpid()}](PLA) Error in Planner service: {e}")
            time.sleep(5)  # 发生错误时等待一段时间再重试
