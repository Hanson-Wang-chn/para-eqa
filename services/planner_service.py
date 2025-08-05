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
    planner_stream_in = STREAMS["planner_to_pool"]
    planner_stream_out = STREAMS["pool_to_planner"]
    
    try:
        redis_conn.xgroup_create(planner_stream_out, "planner_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}](PLA) Planner response group already exists: {e}")
    
    while True:
        # 每次循环生成新的请求ID
        request_id = str(uuid.uuid4())
        request = {
            "request_id": request_id
        }
        
        # 发送请求
        redis_conn.xadd(planner_stream_in, {"data": json.dumps(request)})
        # FIXME: 调试结束后取消注释
        # logging.info(f"[{os.getpid()}](PLA) Sent question selection request: {request_id}")
        
        # 在一个内部循环中等待正确的响应
        wait_start_time = time.time()
        while time.time() - wait_start_time < 1:
            responses = redis_conn.xreadgroup(
                "planner_group", "planner_worker", 
                {planner_stream_out: '>'}, 
                count=1, block=100
            )
            
            if responses:
                for _, msg_list in responses:
                    for msg_id, data in msg_list:
                        response = json.loads(data.get('data', '{}'))
                        response_request_id = response.get('request_id')
                        
                        # 确认消息已处理
                        redis_conn.xack(planner_stream_out, "planner_group", msg_id)
                        
                        # 检查是否是我们的请求的响应
                        if response_request_id == request_id:
                            status = response.get('status')
                            question = response.get('question')
                            
                            if status == "success" and question:
                                logging.info(f"[{os.getpid()}](PLA) Received question {question['id']} from Question Pool")
                                return question
                            else:
                                # FIXME: 调试结束后取消注释
                                # logging.info(f"[{os.getpid()}](PLA) No available questions from Question Pool")
                                break 
            
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
    
    # 主循环
    while True:
        try:
            para_eqa = ParaEQA(config)
            question = select_question(redis_conn)
            para_eqa.run(question, question["id"])
                
        except Exception as e:
            logging.error(f"[{os.getpid()}](PLA) Error in Planner service: {e}")
            time.sleep(5)  # 发生错误时等待一段时间再重试
