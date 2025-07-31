# service/generator_service.py

"""
TODO:
阶段1：读取并处理一组问题
- 在发送每一组的第一个问题之前，先在Redis中通过status存储该问题组的参数（由planner_service.py读取）
    - 直接写入"scene""floor""angle""init_pts""init_rotation"等参数
    - 使用uuid为每一个问题创建"question_id"，然后把所有的{"question_id": correct_answer}存入Redis中
    - 按照一定的时序逻辑发送问题(包括"question_id""description"两个键值对)给parser_service.py

阶段2：读取多组问题（整个数据集）
"""

import os
import json
import time
import logging
import uuid

from common.redis_client import get_redis_connection, STREAMS

def run(config: dict):
    """
    Generator Service 的主运行函数。
    负责按照配置的规则向系统发送问题。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "results")
    logs_dir = os.path.join(parent_dir, "generator_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "generator.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # 硬编码的问题列表
    initial_questions = [
        {"description": "How many floors are there in the house?"},
        {"description": "Is the balcony on the second floor?"},
        {"description": "What is the color of the vase in the balcony?"},
        {"description": "Is the light turned on in the living room?"},
        {"description": "Is there any milk on the table in the dining room?"}
    ]
    
    followup_questions = [
        {"description": "Where can you find a cell phone?"},
        {"description": "How may cushions are there on the sofa in the living room?"},
        {"description": "Is there obvious dirt in my house?"}
    ]
    
    # 连接 Redis
    redis_conn = get_redis_connection(config)
    stream_name = STREAMS["new_questions"]
    
    logging.info(f"[{os.getpid()}] Generator service started.")
    logging.info(f"[{os.getpid()}] Will send {len(initial_questions)} initial questions immediately.")
    logging.info(f"[{os.getpid()}] Will send {len(followup_questions)} follow-up questions every 3 minutes.")
    
    # 初始化计数器
    total_sent = 0
    
    # 发送所有初始问题
    for question in initial_questions:
        # 创建 Question 元数据
        question["id"] = str(uuid.uuid4())
        
        # 发送问题到 new_questions 流
        redis_conn.xadd(stream_name, {"data": json.dumps(question)})
        total_sent += 1
        logging.info(f"[{os.getpid()}] 已发送初始问题 {total_sent}/{len(initial_questions)}: '{question['description']}'")
    
    # # 设置后续问题的间隔时间（3分钟）
    # followup_interval = 180  # 3分钟 = 180秒
    
    # try:
    #     # 循环发送后续问题
    #     for i, question in enumerate(followup_questions):
    #         # 等待指定的间隔时间
    #         logging.info(f"[{os.getpid()}] 等待 {followup_interval} 秒后发送后续问题...")
    #         time.sleep(followup_interval)
            
    #         # 发送问题到 new_questions 流
    #         redis_conn.xadd(stream_name, {"data": json.dumps(question)})
    #         logging.info(f"[{os.getpid()}] 已发送后续问题 {i+1}/{len(followup_questions)}: '{question['description']}'")
        
    #     logging.info(f"[{os.getpid()}] 所有问题已发送完毕，共 {len(initial_questions) + len(followup_questions)} 个问题")
        
    #     # 保持进程运行，直到被终止
    #     while True:
    #         time.sleep(3600)  # 睡眠1小时，保持进程活跃
            
    # except KeyboardInterrupt:
    #     logging.info(f"[{os.getpid()}] Generator service received shutdown signal")
    # except Exception as e:
    #     logging.error(f"[{os.getpid()}] Generator service encountered an error: {e}")
