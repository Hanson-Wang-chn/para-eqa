# services/question_pool_service.py

import os
import json
import time
import logging

from common.redis_client import get_redis_connection, STREAMS, KEY_PREFIXES, PUBSUB_CHANNELS
from utils.updater import Updater

def run(config: dict):
    """
    Question Pool Service 的主运行函数。
    负责维护问题池、更新问题依赖关系及状态、响应添加问题和完成问题的请求。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "results")
    logs_dir = os.path.join(parent_dir, "question_pool_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "question_pool.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # 初始化 Updater 实例
    updater = Updater(config)
    
    # 连接 Redis
    redis_conn = get_redis_connection(config)
    
    # 设置消费者组，用于接收请求
    finishing_stream = STREAMS["finishing_to_pool"]
    stopping_stream = STREAMS["stopping_to_pool"]
    selector_stream = STREAMS["pool_to_selector"]
    
    # 创建消费者组
    try:
        redis_conn.xgroup_create(finishing_stream, "pool_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Finishing consumer group already exists: {e}")
        
    try:
        redis_conn.xgroup_create(stopping_stream, "pool_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Stopping consumer group already exists: {e}")
    
    logging.info(f"[{os.getpid()}] Question Pool service started.")
    
    while True:
        try:
            # 1. 优先检查来自 Finishing Module 的添加问题请求
            finishing_msgs = redis_conn.xreadgroup(
                "pool_group", "pool_worker", 
                {finishing_stream: '>'}, 
                count=1, block=100
            )
            
            if finishing_msgs:
                for _, msg_list in finishing_msgs:
                    for msg_id, data in msg_list:
                        question_data = json.loads(data.get('data', '{}'))
                        logging.info(f"[{os.getpid()}] Received add question request: {question_data['id']}")
                        
                        # 调用 Updater 添加问题
                        try:
                            updater.add_question(question_data)
                            logging.info(f"[{os.getpid()}] Question {question_data['id']} added successfully")
                            
                            # 确认消息已处理
                            redis_conn.xack(finishing_stream, "pool_group", msg_id)
                            
                            # 向 Selector 发送更新后的 buffer
                            send_buffer_to_selector(redis_conn, updater, selector_stream)
                            
                        except Exception as e:
                            logging.error(f"[{os.getpid()}] Error adding question: {e}")
                            # 消息处理失败，仍然确认，避免反复尝试失败的消息
                            redis_conn.xack(finishing_stream, "pool_group", msg_id)
                
                # 处理了添加请求后，继续下一轮循环，优先处理新请求
                continue
            
            # 2. 检查来自 Stopping Module 的完成问题请求
            stopping_msgs = redis_conn.xreadgroup(
                "pool_group", "pool_worker", 
                {stopping_stream: '>'}, 
                count=1, block=100
            )
            
            if stopping_msgs:
                for _, msg_list in stopping_msgs:
                    for msg_id, data in msg_list:
                        question_data = json.loads(data.get('data', '{}'))
                        logging.info(f"[{os.getpid()}] Received complete question request: {question_data['id']}")
                        
                        # 调用 Updater 标记问题已完成
                        try:
                            updater.complete_question(question_data)
                            logging.info(f"[{os.getpid()}] Question {question_data['id']} marked as completed")
                            
                            # 确认消息已处理
                            redis_conn.xack(stopping_stream, "pool_group", msg_id)
                            
                            # 向 Selector 发送更新后的 buffer
                            send_buffer_to_selector(redis_conn, updater, selector_stream)
                            
                        except Exception as e:
                            logging.error(f"[{os.getpid()}] Error completing question: {e}")
                            # 消息处理失败，仍然确认，避免反复尝试失败的消息
                            redis_conn.xack(stopping_stream, "pool_group", msg_id)
            
            # 如果没有消息，稍微休眠以减少CPU使用
            if not finishing_msgs and not stopping_msgs:
                time.sleep(0.1)
                
        except Exception as e:
            logging.error(f"[{os.getpid()}] Unexpected error in Question Pool service: {e}")
            time.sleep(1)  # 遇到意外错误，短暂休眠后继续


def send_buffer_to_selector(redis_conn, updater, selector_stream):
    """
    向 Selector 发送当前 buffer 的完整副本
    """
    try:
        # 获取 buffer 中的问题和 DAG
        buffer_data = {
            "buffer": updater.buffer.get_buffer(),
            "dag": updater.buffer.get_dag(),
        }
        
        # 转换为 JSON 字符串并发送
        redis_conn.xadd(selector_stream, {"data": json.dumps(buffer_data)})
        
        # 发布通知，告知 Selector 问题池已更新
        # redis_conn.publish(PUBSUB_CHANNELS['pool_changed'], "updated")
        
    except Exception as e:
        logging.error(f"Error sending buffer to selector: {e}")
