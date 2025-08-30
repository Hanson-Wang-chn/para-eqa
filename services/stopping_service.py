# services/stopping_service.py

import os
import uuid
import json
import time
import logging

from utils.image_processor import decode_image
from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.get_confidence import get_confidence, get_tryout_confidence


def run(config: dict):
    """
    Main running function for Stopping Service.
    Responsible for determining whether to stop exploration and routing questions to appropriate services.
    """
    # Setup logging
    parent_dir = config.get("output_parent_dir", "logs")
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
    
    # Read configuration
    use_rag = config.get("memory", {}).get("use_rag", True)
    
    stopping_config = config.get("stopping", {})
    retrieval_num = stopping_config.get("retrieval_num", 5)
    confidence_threshold = stopping_config.get("confidence_threshold", 0.7)
    enable_tryout_answer = stopping_config.get("enable_tryout_answer", False)
    
    # VLM configuration
    prompt_get_confidence = config.get("prompt", {}).get("stopping", {}).get("get_confidence", "")
    prompt_get_tryout_answer = config.get("prompt", {}).get("stopping", {}).get("get_tryout_answer", "")
    prompt_get_tryout_confidence = config.get("prompt", {}).get("stopping", {}).get("get_tryout_confidence", "")
    
    config_vlm = config.get("vlm", {}).get("stopping", {})
    model_name = config_vlm.get("model", "qwen/qwen2.5-vl-72b-instruct")
    server = config_vlm.get("server", "openrouter")
    base_url = config_vlm.get("base_url", None)
    api_key = config_vlm.get("api_key", None)
    
    # Redis initialization
    redis_conn = get_redis_connection(config)
    
    # Stream definitions
    planner_to_stopping_stream = STREAMS.get("planner_to_stopping", "stream:planner_to_stopping")  # Receive requests from Planner
    memory_requests_stream = STREAMS["memory_requests"]     # Send requests to Memory
    memory_responses_stream = STREAMS["memory_responses"]  # Receive responses from Memory
    stopping_to_planner_stream = STREAMS["stopping_to_planner"]  # Send messages to Planner
    to_pool_stream = STREAMS["pool_requests"]  # Send completion requests to Question Pool
    to_answering_stream = STREAMS["to_answering"]  # Send questions to Answering
    
    # Create consumer groups
    group_name = "stopping_group"
    try:
        redis_conn.xgroup_create(planner_to_stopping_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](STO) Stopping group already exists: {e}")
        pass
    
    try:
        redis_conn.xgroup_create(memory_responses_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](STO) Memory response group already exists: {e}")
        pass
    
    logging.info(f"[{os.getpid()}](STO) Stopping service started. Waiting for planner requests...")
    
    # Initialize statistics counters
    stop_count = 0
    continue_count = 0
    
    while True:
        try:
            # Receive messages from Planner
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
                    must_stop = request_data.get('must_stop', False)
                    used_steps = request_data.get('used_steps', 0)
                    
                    question_id = question.get('id')
                    question_desc = question.get('description', '')
                    
                    processed_image = decode_image(planner_image) if planner_image else None
                    
                    logging.info(f"[{os.getpid()}](STO) Received request from Planner: {question_id} - '{question_desc[:40]}...'")
                    
                    confidence = None
                    
                    if use_rag:
                        # 1. Send search request to Memory
                        memory_request_id = str(uuid.uuid4())
                        memory_request = {
                            "id": memory_request_id,
                            "operation": "search",
                            "text": question_desc,
                            "image_data": planner_image, # planner_image can only be one image or None
                            "top_k": retrieval_num
                        }
                        
                        redis_conn.xadd(memory_requests_stream, {"data": json.dumps(memory_request)})
                        logging.info(f"[{os.getpid()}](STO) Sent search request to Memory: {memory_request_id}")
                        
                        # 2. Wait for Memory response
                        memory_response = None
                        wait_start_time = time.time()
                        max_wait_time = 300  # Maximum wait time in seconds

                        while memory_response is None and (time.time() - wait_start_time < max_wait_time):
                            try:
                                # Use block parameter to wait efficiently, read multiple messages at once to improve efficiency
                                responses = redis_conn.xreadgroup(
                                    group_name, "stopping_worker", 
                                    {memory_responses_stream: '>'}, 
                                    count=20, block=100
                                )
                                
                                # Regular logging to monitor long waits
                                if time.time() - wait_start_time > 30:
                                    logging.info(f"[{os.getpid()}](STO) Waiting for Memory response for more than 30 seconds, request ID: {memory_request_id}")
                                    wait_start_time = time.time()  # Reset timer to avoid log spam

                                if not responses:
                                    # Block timeout, no messages read, continue to next loop iteration
                                    continue
                                
                                for stream, message_list in responses:
                                    for memory_msg_id, data in message_list:
                                        try:
                                            resp_data = json.loads(data.get('data', '{}'))
                                            resp_request_id = resp_data.get('request_id')

                                            # Check if this is the expected response
                                            if resp_request_id == memory_request_id:
                                                # This is the response we're waiting for
                                                memory_response = resp_data
                                                
                                                # Acknowledge the target message as processed
                                                redis_conn.xack(memory_responses_stream, group_name, memory_msg_id)
                                                
                                                logging.info(f"[{os.getpid()}](STO) Received matching Memory response, request ID: {memory_request_id}, total wait time: {time.time() - wait_start_time:.2f}s")
                                                
                                                # Response found, break out of loop
                                                break
                                            else:
                                                # Not the response we're waiting for, ignore it
                                                pass

                                        except (json.JSONDecodeError, AttributeError) as e:
                                            logging.warning(f"[{os.getpid()}](STO) Unable to parse or process Memory response message (ID: {memory_msg_id}): {e}. Acknowledging this message to prevent infinite loop.")
                                            # For unparseable messages, should acknowledge to prevent repeated processing
                                            redis_conn.xack(memory_responses_stream, group_name, memory_msg_id)
                                            continue
                                    
                                    if memory_response:
                                        break  # Break out of outer for loop

                            except Exception as e:
                                logging.warning(f"[{os.getpid()}](STO) Error occurred while waiting for Memory response: {e}, retrying in 1 second...")
                                time.sleep(1)
                        
                        # 3.1 Process Memory response, calculate confidence
                        if not memory_response or memory_response.get('status') != 'success':
                            logging.warning(f"[{os.getpid()}](STO) Did not receive valid Memory response or request failed")
                            # Default confidence is 0, indicating need to continue exploration
                            confidence = 0.0 if not must_stop else 1.0
                            memory_data = []
                        
                        else:
                            # Extract memory data
                            memory_data = memory_response.get('data', [])
                            
                            if must_stop:
                                # Force stop exploration, set confidence to 1.0
                                confidence = 1.0
                                logging.info(f"[{os.getpid()}](STO) Question {question_id} forced to stop exploration, confidence set to 1.0")
                            
                            elif not memory_data:
                                if not enable_tryout_answer:
                                    confidence = get_confidence(
                                        question_desc=question_desc, 
                                        image=processed_image,
                                        kb=[],  # No memory data
                                        prompt_get_confidence=prompt_get_confidence,
                                        model_name=model_name,
                                        server=server,
                                        base_url=base_url,
                                        api_key=api_key
                                    )
                            
                                else:
                                    confidence = get_tryout_confidence(
                                        question_desc=question_desc, 
                                        image=processed_image,
                                        kb=[],  # No memory data
                                        prompt_get_tryout_answer=prompt_get_tryout_answer,
                                        prompt_get_tryout_confidence=prompt_get_tryout_confidence,
                                        model_name=model_name,
                                        server=server,
                                        base_url=base_url,
                                        api_key=api_key
                                )
                            
                            else:
                                # Calculate confidence
                                if not enable_tryout_answer:
                                    confidence = get_confidence(
                                        question_desc=question_desc, 
                                        image=processed_image,
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
                                        image=processed_image,
                                        kb=memory_data,
                                        prompt_get_tryout_answer=prompt_get_tryout_answer,
                                        prompt_get_tryout_confidence=prompt_get_tryout_confidence,
                                        model_name=model_name,
                                        server=server,
                                        base_url=base_url,
                                        api_key=api_key
                                    )
                                
                                logging.info(f"\n[{os.getpid()}](STO) Question {question_id} confidence: {confidence}\n")
                    
                    # 3.2 Handle case when use_rag == False
                    else:
                        if must_stop:
                            confidence = 1.0
                            memory_data = []
                            logging.info(f"[{os.getpid()}](STO) Question {question_id} forced to stop exploration, confidence set to 1.0")
                        
                        else:
                            if not enable_tryout_answer:
                                confidence = get_confidence(
                                    question_desc=question_desc, 
                                    image=processed_image,
                                    kb=[],  # No memory data
                                    prompt_get_confidence=prompt_get_confidence,
                                    model_name=model_name,
                                    server=server,
                                    base_url=base_url,
                                    api_key=api_key
                                )
                            
                            else:
                                confidence = get_tryout_confidence(
                                    question_desc=question_desc, 
                                    image=processed_image,
                                    kb=[],  # No memory data
                                    prompt_get_tryout_answer=prompt_get_tryout_answer,
                                    prompt_get_tryout_confidence=prompt_get_tryout_confidence,
                                    model_name=model_name,
                                    server=server,
                                    base_url=base_url,
                                    api_key=api_key
                                )
                    
                    
                    # 4. Decide whether to stop exploration based on confidence
                    if confidence >= confidence_threshold:
                        # High confidence, can stop exploration and answer question
                        
                        # Update used_step
                        if used_steps > 0:
                            question["used_steps"] = used_steps
                        
                        # 4.1 Send completion request to Question Pool
                        answer_request = {
                            "request_id": str(uuid.uuid4()),
                            "sender": "stopping",
                            "type": "complete_question",
                            "data": question
                        }
                        redis_conn.xadd(to_pool_stream, {"data": json.dumps(answer_request)})
                        logging.info(f"[{os.getpid()}](STO) Sent question completion request to Question Pool, question: {question_id}")
                        
                        # 4.2 Send stop exploration message to Planner
                        stop_message = {
                            "status": "stop",
                            "question": question,
                            "confidence": confidence
                        }
                        redis_conn.xadd(stopping_to_planner_stream, {"data": json.dumps(stop_message)})
                        logging.info(f"[{os.getpid()}](STO) Sent stop exploration message to Planner, question: {question_id}")
                        
                        # 4.3 Combine Planner's image and Memory's memory data
                        # Create a copy of memory items to avoid modifying original data
                        combined_memory_data = memory_data.copy() if use_rag else []
                        
                        combined_memory_data.append({
                            "id": "planner_image",
                            "text": "Observation from exploration",
                            "image_data": planner_image
                        })
                        
                        # 4.4 Send answering request to Answering service
                        answering_request = {
                            "question": question,
                            "memory_data": combined_memory_data
                        }
                        redis_conn.xadd(to_answering_stream, {"data": json.dumps(answering_request)})
                        logging.info(f"[{os.getpid()}](STO) Sent question to Answering service, question: {question_id}")
                        
                        # Update statistics count
                        stop_count += 1
                    else:
                        # Low confidence, need to continue exploration
                        
                        # Send continue exploration message to Planner
                        continue_message = {
                            "status": "continue",
                            "question": question,
                            "confidence": confidence
                        }
                        redis_conn.xadd(stopping_to_planner_stream, {"data": json.dumps(continue_message)})
                        logging.info(f"[{os.getpid()}](STO) Sent continue exploration message to Planner, question: {question_id}")
                        
                        # Update statistics count
                        continue_count += 1
                    
                    # Update statistics information
                    if "stopping" in STATS_KEYS:
                        pipe = redis_conn.pipeline()
                        pipe.hset(STATS_KEYS["stopping"], "stop_count", stop_count)
                        pipe.hset(STATS_KEYS["stopping"], "continue_count", continue_count)
                        pipe.hset(STATS_KEYS["stopping"], "total", stop_count + continue_count)
                        pipe.execute()
                    
                    # Acknowledge message processing completion
                    redis_conn.xack(planner_to_stopping_stream, group_name, message_id)
        
        except Exception as e:
            logging.exception(f"[{os.getpid()}](STO) Stopping service error occurred: {e}")
            time.sleep(5)  # Wait for a while before retrying when error occurs
