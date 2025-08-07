# services/stopping_service.py

import os
import uuid
import json
import time
import logging

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.get_confidence import get_confidence


def run(config: dict):
    """
    Stopping Service 的主运行函数。
    负责判断是否应该停止探索，并将问题路由到适当的服务。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "results")
    logs_dir = os.path.join(parent_dir, "stopping_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "stopping.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # 读取配置
    stopping_config = config.get("stopping", {})
    retrieval_num = stopping_config.get("retrieval_num", 5)
    confidence_threshold = stopping_config.get("confidence_threshold", 0.7)
    
    # VLM配置
    model_api = config.get("vlm", {}).get("vlm_stopping", "qwen/qwen2.5-vl-72b-instruct")
    use_openrouter = config.get("vlm", {}).get("use_openrouter", False)
    prompt_get_confidence = config.get("prompt", {}).get("stopping", {}).get("get_confidence", "")
    
    # Redis初始化
    redis_conn = get_redis_connection(config)
    
    # 流定义
    planner_to_stopping_stream = STREAMS.get("planner_to_stopping", "stream:planner_to_stopping")  # 从Planner接收请求
    memory_requests_stream = STREAMS["memory_requests"]     # 向Memory发送请求
    memory_responses_stream = STREAMS["memory_responses"]  # 从Memory接收响应
    stopping_to_planner_stream = STREAMS["stopping_to_planner"]  # 向Planner发送消息
    stopping_to_pool_stream = STREAMS["stopping_to_pool"]  # 向Question Pool发送完成请求
    to_answering_stream = STREAMS["to_answering"]  # 向Answering发送问题
    
    # 创建消费者组
    group_name = "stopping_group"
    try:
        redis_conn.xgroup_create(planner_to_stopping_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}](STO) Stopping group already exists: {e}")
    
    try:
        redis_conn.xgroup_create(memory_responses_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}](STO) Memory response group already exists: {e}")
    
    logging.info(f"[{os.getpid()}](STO) Stopping service started. Waiting for planner requests...")
    
    # 初始化统计计数器
    stop_count = 0
    continue_count = 0
    
    while True:
        try:
            # 从Planner接收消息
            messages = redis_conn.xreadgroup(
                group_name, "stopping_worker", 
                {planner_to_stopping_stream: '>'}, 
                count=1, block=None
            )
            
            if not messages:
                time.sleep(0.01)
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    request_data = json.loads(data.get('data', '{}'))
                    question = request_data.get('question', {})
                    planner_image = request_data.get('image', '')
                    
                    question_id = question.get('id')
                    question_desc = question.get('description', '')
                    
                    logging.info(f"[{os.getpid()}](STO) 收到来自Planner的请求: {question_id} - '{question_desc[:40]}...'")
                    
                    # 1. 向Memory发送搜索请求
                    memory_request_id = str(uuid.uuid4())
                    memory_request = {
                        "id": memory_request_id,
                        "operation": "search",
                        "text": question_desc,
                        "image_data": planner_image, # planner_image只能是一张图片或None
                        "top_k": retrieval_num
                    }
                    
                    redis_conn.xadd(memory_requests_stream, {"data": json.dumps(memory_request)})
                    logging.info(f"[{os.getpid()}](STO) 向Memory发送搜索请求: {memory_request_id}")
                    
                    # 2. 等待Memory响应
                    memory_response = None
                    wait_start_time = time.time()
                    max_wait_time = 300  # 最长等待时间，单位秒

                    while memory_response is None and (time.time() - wait_start_time < max_wait_time):
                        try:
                            # 使用block参数高效等待，一次读取多条消息以提高效率
                            responses = redis_conn.xreadgroup(
                                group_name, "stopping_worker", 
                                {memory_responses_stream: '>'}, 
                                count=20, block=100
                            )
                            
                            # 定期日志，监控长时间等待
                            if time.time() - wait_start_time > 30:
                                logging.info(f"[{os.getpid()}](STO) 已等待Memory响应超过30秒，请求ID: {memory_request_id}")
                                wait_start_time = time.time()  # 重置计时器，避免日志刷屏

                            if not responses:
                                # block超时，没有读到任何消息，继续下一次循环等待
                                continue
                            
                            for stream, message_list in responses:
                                for memory_msg_id, data in message_list:
                                    try:
                                        resp_data = json.loads(data.get('data', '{}'))
                                        resp_request_id = resp_data.get('request_id')

                                        # 检查是否是我们期望的响应
                                        if resp_request_id == memory_request_id:
                                            # 是我们等待的响应
                                            memory_response = resp_data
                                            
                                            # 确认目标消息已处理
                                            redis_conn.xack(memory_responses_stream, group_name, memory_msg_id)
                                            
                                            logging.info(f"[{os.getpid()}](STO) 收到匹配的Memory响应，请求ID: {memory_request_id}，总等待时间: {time.time() - wait_start_time:.2f}秒")
                                            
                                            # 已找到响应，跳出循环
                                            break
                                        else:
                                            # 不是我们等待的响应，忽略它
                                            pass

                                    except (json.JSONDecodeError, AttributeError) as e:
                                        logging.warning(f"[{os.getpid()}](STO) 无法解析或处理Memory响应消息 (ID: {memory_msg_id}): {e}。确认此消息以防死循环。")
                                        # 对于无法解析的消息，应该确认，防止反复处理
                                        redis_conn.xack(memory_responses_stream, group_name, memory_msg_id)
                                        continue
                                
                                if memory_response:
                                    break  # 跳出外层for循环

                        except Exception as e:
                            logging.warning(f"[{os.getpid()}](STO) 等待Memory响应时发生错误: {e}，1秒后重试...")
                            time.sleep(1)
                    
                    # 3. 处理Memory响应，计算置信度
                    if not memory_response or memory_response.get('status') != 'success':
                        logging.warning(f"[{os.getpid()}](STO) 未收到有效Memory响应或请求失败")
                        # 默认置信度为0，表示需要继续探索
                        confidence = 0.0
                        memory_data = []
                    else:
                        # 提取记忆数据
                        memory_data = memory_response.get('data', [])
                        
                        if not memory_data:
                            # 没有相关记忆，置信度为0
                            confidence = 0.0
                            logging.info(f"[{os.getpid()}](STO) 问题 {question_id} 没有相关记忆，置信度设为0")
                        else:
                            # 计算置信度
                            confidence = get_confidence(
                                question_desc, 
                                memory_data,
                                prompt_get_confidence,
                                model_api,
                                use_openrouter
                            )
                            
                            logging.info(f"\n[{os.getpid()}](STO) 问题 {question_id} 置信度: {confidence}\n")
                    
                    # 4. 根据置信度决定是否停止探索
                    if confidence >= confidence_threshold:
                        # 置信度高，可以停止探索并回答问题
                        
                        # 4.1 向Planner发送停止探索的消息
                        stop_message = {
                            "status": "stop",
                            "question": question,
                            "confidence": confidence
                        }
                        redis_conn.xadd(stopping_to_planner_stream, {"data": json.dumps(stop_message)})
                        logging.info(f"[{os.getpid()}](STO) 已向Planner发送停止探索消息，问题: {question_id}")
                        
                        # 4.2 向Question Pool发送完成问题的请求
                        redis_conn.xadd(stopping_to_pool_stream, {"data": json.dumps(question)})
                        logging.info(f"[{os.getpid()}](STO) 已向Question Pool发送完成问题请求，问题: {question_id}")
                        
                        # 4.3 合并Planner的图像和Memory的记忆数据
                        # 创建记忆项的副本，避免修改原始数据
                        combined_memory_data = memory_data.copy()
                        
                        combined_memory_data.append({
                            "id": "planner_image",
                            "text": "Observation from exploration",
                            "image_data": planner_image
                        })
                        
                        # 4.4 向Answering服务发送回答问题的请求
                        answering_request = {
                            "question": question,
                            "memory_data": combined_memory_data
                        }
                        redis_conn.xadd(to_answering_stream, {"data": json.dumps(answering_request)})
                        logging.info(f"[{os.getpid()}](STO) 已向Answering服务发送问题，问题: {question_id}")
                        
                        # 更新统计计数
                        stop_count += 1
                    else:
                        # 置信度低，需要继续探索
                        
                        # 向Planner发送继续探索的消息
                        continue_message = {
                            "status": "continue",
                            "question": question,
                            "confidence": confidence
                        }
                        redis_conn.xadd(stopping_to_planner_stream, {"data": json.dumps(continue_message)})
                        logging.info(f"[{os.getpid()}](STO) 已向Planner发送继续探索消息，问题: {question_id}")
                        
                        # 更新统计计数
                        continue_count += 1
                    
                    # 更新统计信息
                    if "stopping" in STATS_KEYS:
                        pipe = redis_conn.pipeline()
                        pipe.hset(STATS_KEYS["stopping"], "stop_count", stop_count)
                        pipe.hset(STATS_KEYS["stopping"], "continue_count", continue_count)
                        pipe.hset(STATS_KEYS["stopping"], "total", stop_count + continue_count)
                        pipe.execute()
                    
                    # 确认消息处理完毕
                    redis_conn.xack(planner_to_stopping_stream, group_name, message_id)
        
        except Exception as e:
            logging.exception(f"[{os.getpid()}](STO) Stopping service发生错误: {e}")
            time.sleep(5)  # 发生错误时等待一段时间再重试
