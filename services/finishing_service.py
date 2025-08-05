# services/finishing_service.py

import os
import uuid
import json
import time
import logging

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.get_confidence import get_confidence


def run(config: dict):
    """
    Finishing Service 的主运行函数。
    负责判断问题是否可以直接回答，并将问题路由到适当的服务。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "results")
    logs_dir = os.path.join(parent_dir, "finishing_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "finishing.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # 读取配置
    finishing_config = config.get("finishing", {})
    retrieval_num = finishing_config.get("retrieval_num", 5)
    confidence_threshold = finishing_config.get("confidence_threshold", 0.7)
    
    # VLM配置
    model_api = config.get("vlm", {}).get("model_api", "gpt-4.1")
    use_openrouter = config.get("vlm", {}).get("use_openrouter", False)
    prompt_get_confidence = config.get("prompt", {}).get("finishing", {}).get("get_confidence", "")
    
    # Redis初始化
    redis_conn = get_redis_connection(config)
    
    # 流定义
    parser_to_finishing_stream = STREAMS["parser_to_finishing"]  # 从Parser接收问题
    memory_requests_stream = STREAMS["memory_requests"]    # 向Memory发送请求
    memory_responses_stream = STREAMS["memory_responses"]  # 从Memory接收响应
    to_answering_stream = STREAMS["to_answering"]         # 向Answering发送问题
    finishing_to_pool_stream = STREAMS["finishing_to_pool"] # 向Question Pool发送问题
    
    # 创建消费者组
    group_name = "finishing_group"
    try:
        redis_conn.xgroup_create(parser_to_finishing_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}](FIN) Finishing group already exists: {e}")
    
    try:
        redis_conn.xgroup_create(memory_responses_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}](FIN) Memory response group already exists: {e}")
    
    logging.info(f"[{os.getpid()}](FIN) Finishing service started. Waiting for parsed questions...")
    
    # 初始化统计计数器
    answered_count = 0
    forwarded_count = 0
    
    while True:
        try:
            # 从解析好的问题流中读取消息
            messages = redis_conn.xreadgroup(
                group_name, "finishing_worker", 
                {parser_to_finishing_stream: '>'}, 
                count=1, block=None
            )
            
            if not messages:
                time.sleep(0.01)
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    question = json.loads(data.get('data', '{}'))
                    question_id = question.get('id')
                    question_desc = question.get('description', '')
                    
                    logging.info(f"[{os.getpid()}](FIN) 收到问题: {question_id} - '{question_desc[:40]}...'")
                    
                    # 1. 向Memory发送搜索请求
                    memory_request_id = str(uuid.uuid4())
                    memory_request = {
                        "id": memory_request_id,
                        "operation": "search",
                        "text": question_desc,
                        "image_data": None,
                        "top_k": retrieval_num
                    }
                    
                    redis_conn.xadd(memory_requests_stream, {"data": json.dumps(memory_request)})
                    logging.info(f"[{os.getpid()}](FIN) 向Memory发送搜索请求: {memory_request_id}")
                    
                    # 2. 等待Memory响应
                    memory_response = None
                    wait_start_time = time.time()
                    max_wait_time = 300 # 最长等待时间，单位秒

                    while memory_response is None and (time.time() - wait_start_time < max_wait_time):
                        try:
                            # 使用block参数高效等待，一次读取多条消息以提高效率
                            responses = redis_conn.xreadgroup(
                                group_name, "finishing_worker", 
                                {memory_responses_stream: '>'}, 
                                count=20, block=100
                            )
                            
                            # 定期日志，监控长时间等待
                            if time.time() - wait_start_time > 30:
                                logging.info(f"[{os.getpid()}](FIN) 已等待Memory响应超过30秒，请求ID: {memory_request_id}")
                                wait_start_time = time.time() # 重置计时器，避免日志刷屏

                            if not responses:
                                # block超时，没有读到任何消息，继续下一次循环等待
                                continue
                            
                            for stream, message_list in responses:
                                for message_id, data in message_list:
                                    try:
                                        resp_data = json.loads(data.get('data', '{}'))
                                        resp_request_id = resp_data.get('request_id')

                                        # 核心逻辑：检查是否是我们期望的响应
                                        if resp_request_id == memory_request_id:
                                            # 是我们等待的响应！处理它。
                                            memory_response = resp_data
                                            
                                            # 【关键】只有在确认是目标消息后，才进行ack
                                            redis_conn.xack(memory_responses_stream, group_name, message_id)
                                            
                                            logging.info(f"[{os.getpid()}](FIN) 收到匹配的Memory响应，请求ID: {memory_request_id}，总等待时间: {time.time() - wait_start_time:.2f}秒")
                                            
                                            # 已找到响应，跳出所有循环
                                            break 
                                        else:
                                            # 不是我们等待的响应，忽略它。
                                            # 不要ack！让它留在流中给其他消费者处理。
                                            # logging.debug(f"[{os.getpid()}](FIN) 忽略了不匹配的响应，目标为 {memory_request_id}，收到 {resp_request_id}")
                                            pass

                                    except (json.JSONDecodeError, AttributeError) as e:
                                        logging.warning(f"[{os.getpid()}](FIN) 无法解析或处理Memory响应消息 (ID: {message_id}): {e}。将确认此坏消息以防死循环。")
                                        # 对于无法解析的坏消息，应该ack掉，防止它反复阻塞流
                                        redis_conn.xack(memory_responses_stream, group_name, message_id)
                                        continue
                                
                                if memory_response:
                                    break # 跳出外层for循环

                        except Exception as e:
                            logging.warning(f"[{os.getpid()}](FIN) 等待Memory响应时发生Redis错误: {e}，1秒后重试...")
                            time.sleep(1)
                    
                    # 3. 处理Memory响应，计算置信度
                    if not memory_response or memory_response.get('status') != 'success':
                        logging.warning(f"[{os.getpid()}](FIN) 未收到有效Memory响应或请求失败")
                        # 默认转发到Question Pool
                        redis_conn.xadd(finishing_to_pool_stream, {"data": json.dumps(question)})
                        forwarded_count += 1
                        logging.info(f"[{os.getpid()}](FIN) 问题 {question_id} 转发到Question Pool")
                    else:
                        # 提取记忆数据
                        memory_data = memory_response.get('data', [])
                        
                        if not memory_data:
                            # 没有相关记忆，转发到Question Pool
                            redis_conn.xadd(finishing_to_pool_stream, {"data": json.dumps(question)})
                            forwarded_count += 1
                            logging.info(f"[{os.getpid()}](FIN) 问题 {question_id} 没有相关记忆，转发到Question Pool")
                        else:
                            # 计算置信度
                            confidence = get_confidence(
                                question_desc, 
                                memory_data,
                                prompt_get_confidence,
                                model_api,
                                use_openrouter
                            )
                            
                            logging.info(f"[{os.getpid()}](FIN) 问题 {question_id} 置信度: {confidence}")
                            
                            # 根据置信度决定去向
                            if confidence > confidence_threshold:
                                # 置信度高，发送到Answering服务
                                answering_request = {
                                    "question": question,
                                    "memory_data": memory_data
                                }
                                redis_conn.xadd(to_answering_stream, {"data": json.dumps(answering_request)})
                                answered_count += 1
                                logging.info(f"[{os.getpid()}](FIN) 问题 {question_id} 发送到Answering服务")
                            else:
                                # 置信度低，发送到Question Pool
                                redis_conn.xadd(finishing_to_pool_stream, {"data": json.dumps(question)})
                                forwarded_count += 1
                                logging.info(f"[{os.getpid()}](FIN) 问题 {question_id} 置信度不足，转发到Question Pool")
                    
                    # 更新统计信息
                    pipe = redis_conn.pipeline()
                    pipe.hset(STATS_KEYS["finishing"], "answered", answered_count)
                    pipe.hset(STATS_KEYS["finishing"], "forwarded", forwarded_count)
                    pipe.hset(STATS_KEYS["finishing"], "total", answered_count + forwarded_count)
                    pipe.execute()
                    
                    # 确认消息处理完毕
                    redis_conn.xack(parser_to_finishing_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"[{os.getpid()}](FIN) Finishing service发生错误: {e}")
            time.sleep(5)  # 发生错误时等待一段时间再重试
