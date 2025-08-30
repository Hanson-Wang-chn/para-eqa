# services/finishing_service.py

import os
import uuid
import json
import time
import logging

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.get_confidence import get_confidence, get_tryout_confidence


def run(config: dict):
    """
    Main function of Finishing Service.
    Responsible for determining whether questions can be answered directly and routing questions to appropriate services.
    """
    # Set up logging
    parent_dir = config.get("output_parent_dir", "logs")
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
    
    # Read configuration
    use_parallel = config.get("use_parallel", True)
    direct_answer = config.get("direct_answer", True)
    
    finishing_config = config.get("finishing", {})
    retrieval_num = finishing_config.get("retrieval_num", 5)
    confidence_threshold = finishing_config.get("confidence_threshold", 0.7)
    enable_tryout_answer = finishing_config.get("enable_tryout_answer", False)
    
    # VLM configuration
    prompt_get_confidence = config.get("prompt", {}).get("finishing", {}).get("get_confidence", "")
    prompt_get_tryout_answer = config.get("prompt", {}).get("stopping", {}).get("get_tryout_answer", "")
    prompt_get_tryout_confidence = config.get("prompt", {}).get("stopping", {}).get("get_tryout_confidence", "")
    
    config_vlm = config.get("vlm", {}).get("finishing", {})
    model_name = config_vlm.get("model", "qwen/qwen2.5-vl-72b-instruct")
    server = config_vlm.get("server", "openrouter")
    base_url = config_vlm.get("base_url", None)
    api_key = config_vlm.get("api_key", None)
    
    # Redis initialization
    redis_conn = get_redis_connection(config)
    
    # Stream definitions
    parser_to_finishing_stream = STREAMS["parser_to_finishing"]  # Receive questions from Parser
    memory_requests_stream = STREAMS["memory_requests"]    # Send requests to Memory
    memory_responses_stream = STREAMS["memory_responses"]  # Receive responses from Memory
    to_answering_stream = STREAMS["to_answering"]         # Send questions to Answering
    to_pool_stream = STREAMS["pool_requests"] # Send questions to Question Pool
    
    # Create consumer groups
    group_name = "finishing_group"
    try:
        redis_conn.xgroup_create(parser_to_finishing_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](FIN) Finishing group already exists: {e}")
        pass
    
    try:
        redis_conn.xgroup_create(memory_responses_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](FIN) Memory response group already exists: {e}")
        pass
    
    logging.info(f"[{os.getpid()}](FIN) Finishing service started. Waiting for parsed questions...")
    
    # Initialize statistics counters
    answered_count = 0
    forwarded_count = 0
    
    while True:
        try:
            # Read messages from parsed questions stream
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
                    
                    logging.info(f"[{os.getpid()}](FIN) Received question: {question_id} - '{question_desc[:40]}...'")
                    
                    # When use_parallel == False or direct_answer == False, forward directly to Question Pool
                    if not use_parallel or not direct_answer:
                        request_id = str(uuid.uuid4())
                        request = {
                            "request_id": request_id,
                            "sender": "finishing",
                            "type": "add_question",
                            "data": question
                        }
                        
                        redis_conn.xadd(to_pool_stream, {"data": json.dumps(request)})
                        forwarded_count += 1
                        logging.info(f"[{os.getpid()}](FIN) Question {question_id} forwarded to Question Pool")
                        
                        # Before skipping subsequent processing, first update statistics and confirm message is processed (xack)
                        pipe = redis_conn.pipeline()
                        pipe.hset(STATS_KEYS["finishing"], "answered", answered_count)
                        pipe.hset(STATS_KEYS["finishing"], "forwarded", forwarded_count)
                        pipe.hset(STATS_KEYS["finishing"], "total", answered_count + forwarded_count)
                        pipe.execute()
                        
                        redis_conn.xack(parser_to_finishing_stream, group_name, message_id)
                        continue
                    
                    # 1. Send search request to Memory
                    memory_request_id = str(uuid.uuid4())
                    memory_request = {
                        "id": memory_request_id,
                        "operation": "search",
                        "text": question_desc,
                        "image_data": None,
                        "top_k": retrieval_num
                    }
                    
                    redis_conn.xadd(memory_requests_stream, {"data": json.dumps(memory_request)})
                    logging.info(f"[{os.getpid()}](FIN) Sent search request to Memory: {memory_request_id}")
                    
                    # 2. Wait for Memory response
                    memory_response = None
                    wait_start_time = time.time()
                    max_wait_time = 300 # Maximum wait time in seconds

                    while memory_response is None and (time.time() - wait_start_time < max_wait_time):
                        try:
                            # Use block parameter for efficient waiting, read multiple messages at once to improve efficiency
                            responses = redis_conn.xreadgroup(
                                group_name, "finishing_worker", 
                                {memory_responses_stream: '>'}, 
                                count=20, block=100
                            )
                            
                            # Periodic logging to monitor long waits
                            if time.time() - wait_start_time > 30:
                                logging.info(f"[{os.getpid()}](FIN) Waiting for Memory response over 30 seconds, request ID: {memory_request_id}")
                                wait_start_time = time.time() # Reset timer to avoid log spam

                            if not responses:
                                # Block timeout, no messages read, continue to next loop iteration
                                continue
                            
                            for stream, message_list in responses:
                                for message_id, data in message_list:
                                    try:
                                        resp_data = json.loads(data.get('data', '{}'))
                                        resp_request_id = resp_data.get('request_id')

                                        # Core logic: check if this is the expected response
                                        if resp_request_id == memory_request_id:
                                            # This is the response we're waiting for! Process it.
                                            memory_response = resp_data
                                            
                                            # [KEY] Only ack after confirming this is the target message
                                            redis_conn.xack(memory_responses_stream, group_name, message_id)
                                            
                                            logging.info(f"[{os.getpid()}](FIN) Received matching Memory response, request ID: {memory_request_id}, total wait time: {time.time() - wait_start_time:.2f} seconds")
                                            
                                            # Found response, break out of all loops
                                            break 
                                        else:
                                            # Not the response we're waiting for, ignore it.
                                            # Don't ack! Let it remain in the stream for other consumers to process.
                                            # logging.debug(f"[{os.getpid()}](FIN) Ignored non-matching response, expecting {memory_request_id}, received {resp_request_id}")
                                            pass

                                    except (json.JSONDecodeError, AttributeError) as e:
                                        logging.warning(f"[{os.getpid()}](FIN) Cannot parse or process Memory response message (ID: {message_id}): {e}. Will ack this bad message to prevent deadlock.")
                                        # For unparseable bad messages, should ack to prevent them from repeatedly blocking the stream
                                        redis_conn.xack(memory_responses_stream, group_name, message_id)
                                        continue
                                
                                if memory_response:
                                    break # Break out of outer for loop

                        except Exception as e:
                            logging.warning(f"[{os.getpid()}](FIN) Redis error while waiting for Memory response: {e}, retrying in 1 second...")
                            time.sleep(1)
                    
                    # 3. Process Memory response, calculate confidence
                    if not memory_response or memory_response.get('status') != 'success':
                        logging.warning(f"[{os.getpid()}](FIN) Did not receive valid Memory response or request failed")
                        
                        # Default forward to Question Pool
                        request_id = str(uuid.uuid4())
                        request = {
                            "request_id": request_id,
                            "sender": "finishing",
                            "type": "add_question",
                            "data": question
                        }
                            
                        redis_conn.xadd(to_pool_stream, {"data": json.dumps(request)})
                        forwarded_count += 1
                        logging.info(f"[{os.getpid()}](FIN) Question {question_id} forwarded to Question Pool")
                    
                    else:
                        # Extract memory data
                        memory_data = memory_response.get('data', [])
                        
                        if not memory_data:
                            # No relevant memory, forward to Question Pool
                            request_id = str(uuid.uuid4())
                            request = {
                                "request_id": request_id,
                                "sender": "finishing",
                                "type": "add_question",
                                "data": question
                            }
                            
                            redis_conn.xadd(to_pool_stream, {"data": json.dumps(request)})
                            forwarded_count += 1
                            logging.info(f"[{os.getpid()}](FIN) Question {question_id} has no relevant memory, forwarded to Question Pool")
                        
                        else:
                            # Calculate confidence
                            if not enable_tryout_answer:
                                confidence = get_confidence(
                                    question_desc=question_desc, 
                                    image=None,
                                    kb=memory_data,
                                    prompt_get_confidence=prompt_get_confidence,
                                    model_name=model_name,
                                    server=server,
                                    base_url=base_url,
                                    api_key=api_key
                                )
                            
                            else:
                                confidence = get_tryout_confidence(
                                    question_desc=question_desc, 
                                    image=None,
                                    kb=memory_data,
                                    prompt_get_tryout_answer=prompt_get_tryout_answer,
                                    prompt_get_tryout_confidence=prompt_get_tryout_confidence,
                                    model_name=model_name,
                                    server=server,
                                    base_url=base_url,
                                    api_key=api_key
                                )
                            
                            logging.info(f"[{os.getpid()}](FIN) Question {question_id} confidence: {confidence}")
                            
                            # Decide destination based on confidence
                            if confidence >= confidence_threshold:
                                # High confidence, send to Answering service
                                
                                # Set question start processing time
                                if "time" not in question:
                                    question["time"] = {}
                                question["time"]["start"] = time.time()
                                
                                answering_request = {
                                    "question": question,
                                    "memory_data": memory_data
                                }
                                redis_conn.xadd(to_answering_stream, {"data": json.dumps(answering_request)})
                                answered_count += 1
                                logging.info(f"[{os.getpid()}](FIN) Question {question_id} sent to Answering service")
                            
                            else:
                                # Low confidence, send to Question Pool
                                request_id = str(uuid.uuid4())
                                request = {
                                    "request_id": request_id,
                                    "sender": "finishing",
                                    "type": "add_question",
                                    "data": question
                                }
                                
                                redis_conn.xadd(to_pool_stream, {"data": json.dumps(request)})
                                forwarded_count += 1
                                
                                logging.info(f"[{os.getpid()}](FIN) Question {question_id} insufficient confidence, forwarded to Question Pool")
                    
                    # Update statistics
                    pipe = redis_conn.pipeline()
                    pipe.hset(STATS_KEYS["finishing"], "answered", answered_count)
                    pipe.hset(STATS_KEYS["finishing"], "forwarded", forwarded_count)
                    pipe.hset(STATS_KEYS["finishing"], "total", answered_count + forwarded_count)
                    pipe.execute()
                    
                    # Confirm message processing completed
                    redis_conn.xack(parser_to_finishing_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"[{os.getpid()}](FIN) Finishing service error: {e}")
            time.sleep(5)  # Wait for a while before retrying when error
