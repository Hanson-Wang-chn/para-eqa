# services/planner_service.py

import os
import json
import time
import logging
import uuid

from common.redis_client import get_redis_connection, STREAMS
from utils.para_eqa import ParaEQA


def select_question(redis_conn):
    """
    Request a highest priority question from Question Pool
    
    Args:
        redis_conn: Redis connection object
        
    Returns:
        dict: Question object, returns None if no available questions
    """
    # Create consumer group (if not exists)
    pool_responses_stream = STREAMS["pool_responses"]
    
    try:
        redis_conn.xgroup_create(pool_responses_stream, "planner_group", id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PLA) Planner response group already exists: {e}")
        pass
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Create request
    request = {
        "request_id": request_id,
        "sender": "planner",
        "type": "select_question",
        "data": {}
    }
    
    # Send request - only send once
    redis_conn.xadd(STREAMS["pool_requests"], {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](PLA) Sent question selection request: {request_id}")
    
    # Infinite loop waiting for response, no longer repeatedly sending requests
    while True:
        try:
            responses = redis_conn.xreadgroup(
                "planner_group", "planner_worker", 
                {pool_responses_stream: '>'}, 
                count=20, block=100
            )
            
            if not responses:
                continue
                
            for _, msg_list in responses:
                for msg_id, data in msg_list:
                    try:
                        response = json.loads(data.get('data', '{}'))
                        response_request_id = response.get('request_id')
                        
                        # Check if this is the response we requested
                        if response_request_id == request_id and response.get('type') == 'question_selected':
                            # Acknowledge the matching message
                            redis_conn.xack(pool_responses_stream, "planner_group", msg_id)
                            
                            if response.get('status') == "success" and response.get('data'):
                                logging.info(f"[{os.getpid()}](PLA) Received question {response['data']['id']} from Question Pool")
                                return response['data']
                            
                            else:
                                logging.info(f"[{os.getpid()}](PLA) No available questions from Question Pool")
                                return None
                        
                        else:
                            # Not our response, don't acknowledge the message
                            logging.debug(f"[{os.getpid()}](PLA) Received non-target response: {response_request_id}, waiting for: {request_id}")
                    
                    except Exception as e:
                        logging.error(f"[{os.getpid()}](PLA) Error processing response: {e}")
                        # Handle error but continue waiting for correct response
        
        except Exception as e:
            logging.error(f"[{os.getpid()}](PLA) Error while waiting for response: {e}")
            time.sleep(1)  # Wait briefly after error before continuing


def clear_memory(redis_conn):
    """
    Send request to Memory Service to clear knowledge base
    
    Args:
        redis_conn: Redis connection object
        
    Returns:
        bool: Whether the operation was successful
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Create request
    request = {
        "id": request_id,
        "operation": "clear"
    }
    
    # Send request
    redis_conn.xadd(STREAMS["memory_requests"], {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](PLA) Sent clear knowledge base request to Memory Service: {request_id}")
    
    return True


def run(config: dict):
    """
    Main run function of Planner Service.
    Responsible for planning exploration paths and interacting with environment.
    """
    # Set up logging
    parent_dir = config.get("output_parent_dir", "logs")
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
    
    # Connect to Redis
    redis_conn = get_redis_connection(config)
    
    logging.info(f"[{os.getpid()}](PLA) Planner service started.")
    
    # Instantiate ParaEQA
    para_eqa = ParaEQA(config)
    
    # Main loop
    while True:
        try:
            question = select_question(redis_conn)
            if question is None:
                logging.info(f"[{os.getpid()}](PLA) No available questions, waiting for new questions...")
                time.sleep(1)
                continue
            
            # Process question
            para_eqa.run(question, question["id"])
            
            # Clear memory after processing is complete
            clear_memory(redis_conn)
            logging.info(f"[{os.getpid()}](PLA) Question {question['id']} processing completed, knowledge base cleared")
                
        except Exception as e:
            logging.exception(f"[{os.getpid()}](PLA) Error in Planner service: {e}")
            time.sleep(5)  # Wait for a while before retrying when error occurs
