# services/question_pool_service.py

import os
import json
import time
import logging
import uuid

from common.redis_client import get_redis_connection, STREAMS
from utils.updater import Updater
from utils.get_current_group_id import get_current_group_id


def send_group_completed(redis_conn, group_id, request_id=None):
    """
    向Generator Service发送组完成消息
    """
    msg = {
        "request_id": request_id or str(uuid.uuid4()),
        "status": "success",
        "type": "group_completed",
        "data": {
            "group_id": group_id
        }
    }
    redis_conn.xadd(STREAMS["pool_responses"], {"data": json.dumps(msg)})
    logging.info(f"[{os.getpid()}](QUE) 已发送组完成消息: {group_id}")


def run(config: dict):
    """
    Question Pool Service 的主运行函数。
    负责维护问题池、更新问题依赖关系及状态、响应添加问题和完成问题的请求。
    """
    # 设置日志部分保持不变...
    
    # 初始化 Updater 实例
    updater = Updater(config)
    
    # 连接 Redis
    redis_conn = get_redis_connection(config)
    
    # 设置消费者组，只需要监听一个请求流
    pool_requests_stream = STREAMS["pool_requests"]
    pool_responses_stream = STREAMS["pool_responses"]
    
    # 创建消费者组
    try:
        redis_conn.xgroup_create(pool_requests_stream, "pool_group", id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](QUE) Pool requests consumer group already exists: {e}")
        pass
    
    logging.info(f"[{os.getpid()}](QUE) Question Pool service started.")
    
    while True:
        try:
            # 监听单一请求流
            messages = redis_conn.xreadgroup(
                "pool_group", "pool_worker", 
                {pool_requests_stream: '>'}, 
                count=1, block=100
            )
            
            if not messages:
                time.sleep(0.01)
                continue
            
            for _, msg_list in messages:
                for msg_id, data in msg_list:
                    try:
                        request = json.loads(data.get('data', '{}'))
                        request_id = request.get('request_id', str(uuid.uuid4()))
                        request_type = request.get('type')
                        request_data = request.get('data', {})
                        sender = request.get('sender', 'unknown')
                        
                        logging.info(f"[{os.getpid()}](QUE) 收到请求 {request_id}, 类型: {request_type}, 发送者: {sender}")
                        
                        # 根据请求类型分发处理
                        if request_type == "add_question":
                            # 处理添加问题请求
                            try:
                                updater.add_question(request_data)
                                logging.info(f"[{os.getpid()}](QUE) Question {request_data['id']} added successfully")
                                
                                # 发送成功响应
                                response = {
                                    "request_id": request_id,
                                    "status": "success",
                                    "type": "question_added",
                                    "data": {"id": request_data['id']}
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                
                            except Exception as e:
                                logging.exception(f"[{os.getpid()}](QUE) Error adding question: {e}")
                                # 发送错误响应
                                response = {
                                    "request_id": request_id,
                                    "status": "error",
                                    "type": "question_added",
                                    "error": str(e)
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        elif request_type == "complete_question":
                            # 处理完成问题请求
                            try:
                                updater.complete_question(request_data)
                                logging.info(f"[{os.getpid()}](QUE) Question {request_data['id']} marked as completed")
                                
                                # 发送成功响应
                                response = {
                                    "request_id": request_id,
                                    "status": "success", 
                                    "type": "question_completed",
                                    "data": {"id": request_data['id']}
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                
                            except Exception as e:
                                logging.error(f"[{os.getpid()}](QUE) Error completing question: {e}")
                                # 发送错误响应
                                response = {
                                    "request_id": request_id,
                                    "status": "error",
                                    "type": "question_completed",
                                    "error": str(e)
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        elif request_type == "select_question":
                            # 处理选择问题请求
                            highest_priority_question = updater.get_highest_priority_question()
                            
                            if highest_priority_question:
                                # 更新问题状态为 "in_progress"
                                updater.buffer.set_status(highest_priority_question["id"], "in_progress")
                                
                                # 返回问题
                                response = {
                                    "request_id": request_id,
                                    "status": "success",
                                    "type": "question_selected",
                                    "data": highest_priority_question
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                logging.info(f"[{os.getpid()}](QUE) Sent question {highest_priority_question['id']} to requester")
                            else:
                                # 没有可用问题
                                response = {
                                    "request_id": request_id,
                                    "status": "empty",
                                    "type": "question_selected",
                                    "data": None
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        elif request_type == "answer_question":
                            # 处理添加答案请求
                            try:
                                question_id = request_data.get('id')
                                
                                try:
                                    # 检查buffer中是否已有该问题
                                    existing_question = updater.buffer.get_question_by_id(question_id)
                                    
                                    if existing_question["status"] == "completed":
                                        # 更新问题答案和状态
                                        updater.answer_question(request_data)
                                        logging.info(f"[{os.getpid()}](QUE) Answer added to question {question_id}")

                                        # 检查问题组是否全部完成
                                        group_id = get_current_group_id(redis_conn)
                                        if group_id and updater.is_group_completed(redis_conn, group_id):
                                            send_group_completed(redis_conn, group_id, request_id)
                                        
                                        # 发送成功响应
                                        response = {
                                            "request_id": request_id,
                                            "status": "success",
                                            "type": "answer_added",
                                            "data": {"id": question_id}
                                        }
                                        redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                    else:
                                        # 状态不是completed
                                        msg = f"Question {question_id} status is {existing_question['status']}, not completed"
                                        logging.error(f"[{os.getpid()}](QUE) {msg}")
                                        
                                        response = {
                                            "request_id": request_id,
                                            "status": "error",
                                            "type": "answer_added",
                                            "error": msg
                                        }
                                        redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                        
                                except ValueError:
                                    # 问题不在buffer中，添加新问题
                                    request_data["status"] = "answered"
                                    updater.buffer.write_latest_questions([request_data])
                                    
                                    logging.info(f"[{os.getpid()}](QUE) New answered question {question_id} added to buffer")
                                    
                                    # 发送成功响应
                                    response = {
                                        "request_id": request_id,
                                        "status": "success",
                                        "type": "answer_added",
                                        "data": {"id": question_id}
                                    }
                                    redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                            
                            except Exception as e:
                                logging.error(f"[{os.getpid()}](QUE) Error processing answer: {e}")
                                # 发送错误响应
                                response = {
                                    "request_id": request_id,
                                    "status": "error",
                                    "type": "answer_added",
                                    "error": str(e)
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        elif request_type == "check_group_completed":
                            # 检查组是否完成
                            group_id = request_data.get('group_id')
                            if updater.is_group_completed(redis_conn, group_id):
                                send_group_completed(redis_conn, group_id, request_id)
                            else:
                                response = {
                                    "request_id": request_id,
                                    "status": "success",
                                    "type": "group_status",
                                    "data": {"completed": False, "group_id": group_id}
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        else:
                            # 未知请求类型
                            logging.warning(f"[{os.getpid()}](QUE) Unknown request type: {request_type}")
                            
                            response = {
                                "request_id": request_id,
                                "status": "error",
                                "type": "unknown",
                                "error": f"Unknown request type: {request_type}"
                            }
                            redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                    
                    except Exception as e:
                        logging.error(f"[{os.getpid()}](QUE) Error processing request: {e}")
                    
                    finally:
                        # 无论如何都确认消息已处理
                        redis_conn.xack(pool_requests_stream, "pool_group", msg_id)
            
        except Exception as e:
            logging.error(f"[{os.getpid()}](QUE) Unexpected error in Question Pool service: {e}")
            time.sleep(5)  # 遇到意外错误，短暂休眠后继续
