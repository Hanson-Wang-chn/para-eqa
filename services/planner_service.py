# services/planner_service.py

# TODO: 处理好异步逻辑。问题的初始化信息都在Redis中。第一个问题的初始化信息由generator提供，后续每一个问题的初始化信息由前一个问题提供。为ParaEQA提供初始化信息（初始位置为上一次结束探索的位置）。

import os
import json
import time
import logging
import uuid

from common.redis_client import get_redis_connection, STREAMS


def select_question(redis_conn):
    """
    向Question Pool请求一个优先级最高的问题
    
    Args:
        redis_conn: Redis连接对象
        
    Returns:
        dict: 问题对象，如果没有可用问题则返回None
    """
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "timestamp": time.time()
    }
    
    # 发送请求到Question Pool
    planner_stream_in = STREAMS["planner_to_pool"]
    planner_stream_out = STREAMS["pool_to_planner"]
    
    # 创建消费者组（如果不存在）
    try:
        redis_conn.xgroup_create(planner_stream_out, "planner_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Planner response group already exists: {e}")
    
    # 发送请求
    redis_conn.xadd(planner_stream_in, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}] Sent question selection request: {request_id}")
    
    # 等待响应
    max_wait_time = 300  # 最长等待300秒
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
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
                    
                    # 检查是否是我们的请求的响应
                    if response_request_id == request_id:
                        status = response.get('status')
                        question = response.get('question')
                        
                        # 确认消息已处理
                        redis_conn.xack(planner_stream_out, "planner_group", msg_id)
                        
                        if status == "success" and question:
                            logging.info(f"[{os.getpid()}] Received question {question['id']} from Question Pool")
                            return question
                        
                        else:
                            logging.info(f"[{os.getpid()}] No available questions from Question Pool")
                    
                    else:
                        # 不是我们期望的响应，但仍然确认收到
                        redis_conn.xack(planner_stream_out, "planner_group", msg_id)
        
        # 短暂休眠以减少CPU使用
        time.sleep(1)
    
    logging.warning(f"[{os.getpid()}] Timed out waiting for question selection response")
    return None


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
    
    logging.info(f"[{os.getpid()}] Planner service started.")
    
    # 主循环
    while True:
        try:
            group_info = get_group_info_from_redis()
            para_eqa = ParaEQA(config, group_info)
            
            question = select_question(redis_conn)
            result = para_eqa.run(question)
            add_result_to_redis(result, redis_conn)
                
        except Exception as e:
            logging.error(f"[{os.getpid()}] Error in Planner service: {e}")
            time.sleep(5)  # 发生错误时等待一段时间再重试
