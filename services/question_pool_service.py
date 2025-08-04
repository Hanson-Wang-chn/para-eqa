# services/question_pool_service.py

import os
import json
import time
import logging
import uuid

from common.redis_client import get_redis_connection, STREAMS, GROUP_INFO
from utils.updater import Updater
from utils.get_current_group_id import get_current_group_id


def send_group_completed(redis_conn, group_id):
    """
    向Generator Service发送组完成消息
    """
    msg = {
        "status": "group_completed",
        "group_id": group_id
    }
    pool_to_generator_stream = STREAMS["pool_to_generator"]
    redis_conn.xadd(pool_to_generator_stream, {"data": json.dumps(msg)})
    logging.info(f"[{os.getpid()}] 已向Generator Service发送组完成消息: {group_id}")


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
    planner_stream_in = STREAMS["planner_to_pool"]
    planner_stream_out = STREAMS["pool_to_planner"]
    answering_stream = STREAMS["answering_to_pool"]
    
    # 创建消费者组
    try:
        redis_conn.xgroup_create(finishing_stream, "pool_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Finishing consumer group already exists: {e}")
        
    try:
        redis_conn.xgroup_create(stopping_stream, "pool_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Stopping consumer group already exists: {e}")
        
    try:
        redis_conn.xgroup_create(planner_stream_in, "pool_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Planner consumer group already exists: {e}")
    
    try:
        redis_conn.xgroup_create(answering_stream, "pool_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Answering consumer group already exists: {e}")
    
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
                            
                        except Exception as e:
                            logging.error(f"[{os.getpid()}] Error completing question: {e}")
                            # 消息处理失败，仍然确认，避免反复尝试失败的消息
                            redis_conn.xack(stopping_stream, "pool_group", msg_id)
            
            # 3. 检查来自 Planner 的选择问题请求
            planner_msgs = redis_conn.xreadgroup(
                "pool_group", "pool_worker", 
                {planner_stream_in: '>'}, 
                count=1, block=100
            )
            
            if planner_msgs:
                for _, msg_list in planner_msgs:
                    for msg_id, data in msg_list:
                        planner_request = json.loads(data.get('data', '{}'))
                        request_id = planner_request.get('request_id', str(uuid.uuid4()))
                        logging.info(f"[{os.getpid()}] Received select question request from Planner: {request_id}")
                        
                        # 获取最高优先级的问题
                        highest_priority_question = updater.highest_priority_question
                        
                        # 如果找到了高优先级问题
                        if highest_priority_question:
                            # 更新问题状态为 "in_progress"
                            updater.buffer.set_status(highest_priority_question["id"], "in_progress")
                            
                            # 返回问题到Planner
                            response = {
                                "request_id": request_id,
                                "status": "success",
                                "question": highest_priority_question
                            }
                            redis_conn.xadd(planner_stream_out, {"data": json.dumps(response)})
                            logging.info(f"[{os.getpid()}] Sent question {highest_priority_question['id']} to Planner")
                        else:
                            # 如果没有可用问题，返回空响应
                            response = {
                                "request_id": request_id,
                                "status": "empty",
                                "question": None
                            }
                            redis_conn.xadd(planner_stream_out, {"data": json.dumps(response)})
                            logging.info(f"[{os.getpid()}] No available questions to send to Planner")
                        
                        # 确认消息已处理
                        redis_conn.xack(planner_stream_in, "pool_group", msg_id)
                
                continue
            
            # 4. 检查来自 Answering Module 的回答写入请求
            answering_msgs = redis_conn.xreadgroup(
                "pool_group", "pool_worker", 
                {answering_stream: '>'}, 
                count=1, block=100
            )

            if answering_msgs:
                for _, msg_list in answering_msgs:
                    for msg_id, data in msg_list:
                        try:
                            question_data = json.loads(data.get('data', '{}'))
                            question_id = question_data.get('id')
                            logging.info(f"[{os.getpid()}] Received answer for question: {question_id}")
                            
                            try:
                                # 检查buffer中是否已有该问题
                                existing_question = updater.buffer.get_question_by_id(question_id)
                                
                                # 如果存在，检查状态是否为completed
                                if existing_question["status"] == "completed":
                                    # 更新问题答案和状态
                                    updater.answer_question(question_data)
                                    logging.info(f"[{os.getpid()}] Answer added to question {question_id} and marked as answered")

                                    # 检查问题组是否全部完成
                                    # group_id = redis_conn.get(f"{GROUP_INFO['group_id']}{question_data.get('group_id', '')}")
                                    group_id = get_current_group_id(redis_conn)
                                    if group_id and updater.is_group_completed(redis_conn, group_id):
                                        send_group_completed(redis_conn, group_id)
                                
                                else:
                                    # 状态不是completed，报错
                                    logging.error(f"[{os.getpid()}] Question {question_id} status is {existing_question['status']}, not completed. Cannot add answer.")
                                    
                            except ValueError:
                                # 问题不在buffer中，添加新问题，状态为answered
                                question_data["status"] = "answered"
                                updater.buffer.write_latest_questions([question_data])
                                
                                logging.info(f"[{os.getpid()}] New answered question {question_id} added to buffer")
                            
                        except Exception as e:
                            logging.error(f"[{os.getpid()}] Error processing answer from Answering Module: {e}")
                        
                        # 确认消息已处理
                        redis_conn.xack(answering_stream, "pool_group", msg_id)
                
                continue
            
            # 如果没有消息，稍微休眠以减少CPU使用
            if not finishing_msgs and not stopping_msgs and not planner_msgs and not answering_msgs:
                time.sleep(0.1)
                
        except Exception as e:
            logging.error(f"[{os.getpid()}] Unexpected error in Question Pool service: {e}")
            time.sleep(5)  # 遇到意外错误，短暂休眠后继续

