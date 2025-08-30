# services/answering_service.py

import os
import json
import time
import logging
import uuid

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.vlm_api import VLM_API
from utils.image_processor import decode_image
from utils.get_current_group_id import get_current_group_id


def get_vlm_answer(question, kb, prompt_get_answer, model_name="qwen/qwen2.5-vl-72b-instruct", server="openrouter", base_url=None, api_key=None):
    """
    Generate answer using VLM based on question and memory data
    
    Args:
        question (dict): Question object containing description and other information
        memory_data (list): Data retrieved from memory
        model_api (str): OpenAI model name to use
        
    Returns:
        str: Generated answer
    """
    # Build prompt
    question_desc = question.get('description', '')
    prompt = prompt_get_answer.format(question_desc)
    
    # Instantiate VLM and request answer
    vlm = VLM_API(model_name=model_name, server="openrouter", base_url=base_url, api_key=api_key)
    response = vlm.request_with_retry(image=None, prompt=prompt, kb=kb)[0]
    
    return response.strip()


def run(config: dict):
    """
    Main run function for Answering Service.
    Responsible for receiving questions, generating answers, and sending answers to Question Pool.
    """
    # Setup logging
    parent_dir = config.get("output_parent_dir", "logs")
    logs_dir = os.path.join(parent_dir, "answering_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "answering.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # Read configuration
    answering_config = config.get("answering", {})
    result_dir = answering_config.get("result_dir", "results/answers")
    
    # Ensure result directory exists
    os.makedirs(result_dir, exist_ok=True)

    # VLM configuration
    prompt_get_answer = config.get("prompt", {}).get("answering", {}).get("get_answer", "")
    
    config_vlm = config.get("vlm", {}).get("answering", {})
    model_name = config_vlm.get("model", "qwen/qwen2.5-vl-72b-instruct")
    server = config_vlm.get("server", "openrouter")
    base_url = config_vlm.get("base_url", None)
    api_key = config_vlm.get("api_key", None)
    
    # Redis initialization
    redis_conn = get_redis_connection(config)
    
    # Stream definition
    to_answering_stream = STREAMS["to_answering"]  # Receive questions from Finishing
    to_pool_stream = STREAMS["pool_requests"]  # Send answers to Question Pool
    
    # Create consumer group
    group_name = "answering_group"
    try:
        redis_conn.xgroup_create(to_answering_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](ANS) Answering group already exists: {e}")
        pass
    
    logging.info(f"[{os.getpid()}](ANS) Answering service started. Waiting for questions...")
    
    # Initialize statistics counter
    answered_count = 0
    
    while True:
        try:
            # Read messages from to_answering stream
            messages = redis_conn.xreadgroup(
                group_name, "answering_worker", 
                {to_answering_stream: '>'}, 
                count=1, block=None
            )
            
            if not messages:
                time.sleep(0.01)
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    try:
                        # Parse request data
                        request_data = json.loads(data.get('data', '{}'))
                        question = request_data.get('question', {})
                        memory_data = request_data.get('memory_data', [])
                        
                        question_id = question.get('id')
                        question_desc = question.get('description', '')
                        
                        logging.info(f"[{os.getpid()}](ANS) Received question: {question_id} - '{question_desc[:40]}...'")
                        
                        # What colors are the cushions on the white sofa on the first floor? A) Blue and orange B) Red and green C) Black and gray D) Yellow and pink.
                        
                        # Get answer
                        answer = get_vlm_answer(question, memory_data, prompt_get_answer, model_name, server, base_url, api_key)
                        logging.info(f"[{os.getpid()}](ANS) Generated answer for question {question_id}")
                        
                        # Update question metadata
                        question['answer'] = answer
                        
                        # Set finish time
                        if "time" not in question:
                            question["time"] = {}
                        question["time"]["finish"] = time.time()
                        
                        # Save answer to file
                        try:
                            # Get current group_id
                            group_id = get_current_group_id(redis_conn)
                            
                            if not group_id:
                                logging.error(f"[{os.getpid()}](ANS) Unable to get current group_id, cannot save answer")
                                continue  # Skip saving
                            
                            else:
                                group_id_str = group_id.decode('utf-8') if isinstance(group_id, bytes) else str(group_id)
                                group_specific_result_path = os.path.join(result_dir, f"answers_{group_id_str}.json")
                            
                            os.makedirs(os.path.dirname(group_specific_result_path), exist_ok=True)
                            existing_answers = []
                            try:
                                if os.path.exists(group_specific_result_path):
                                    with open(group_specific_result_path, 'r', encoding='utf-8') as f:
                                        existing_answers = json.load(f)
                                
                                else:
                                    logging.info(f"[{os.getpid()}](ANS) Creating new answer file for group {group_id_str}")
                            
                            except (json.JSONDecodeError, FileNotFoundError):
                                logging.warning(f"[{os.getpid()}](ANS) Unable to read existing answer file, will create new file")
                            
                            # Check if question with same ID already exists
                            for i, existing_answer in enumerate(existing_answers):
                                if existing_answer.get('id') == question_id:
                                    existing_answers[i] = question
                                    break
                            
                            else:
                                existing_answers.append(question)
                            
                            with open(group_specific_result_path, 'w', encoding='utf-8') as f:
                                json.dump(existing_answers, f, ensure_ascii=False, indent=2)
                                
                            logging.info(f"[{os.getpid()}](ANS) Answer for question {question_id} saved to file {os.path.basename(group_specific_result_path)}")
                        
                        except Exception as e:
                            logging.error(f"[{os.getpid()}](ANS) Error saving answer to file: {e}")
                        
                        # Send completed question to Question Pool
                        answer_request = {
                            "request_id": str(uuid.uuid4()),
                            "sender": "answering",
                            "type": "answer_question",
                            "data": question
                        }
                        redis_conn.xadd(to_pool_stream, {"data": json.dumps(answer_request)})
                        logging.info(f"[{os.getpid()}](ANS) Answer for question {question_id} sent to Question Pool")
                        
                        # Update statistics
                        answered_count += 1
                        redis_conn.hset(STATS_KEYS["answering"], "answered", answered_count)
                        
                    except Exception as e:
                        logging.error(f"[{os.getpid()}](ANS) Error processing question: {e}")
                    
                    # Acknowledge message processing complete
                    redis_conn.xack(to_answering_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"[{os.getpid()}](ANS) Answering service error: {e}")
            time.sleep(5)  # Wait for a while before
