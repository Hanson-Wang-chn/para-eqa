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
    Send group completion message to Generator Service
    """
    msg = {
        "request_id": request_id or str(uuid.uuid4()),
        "status": "success",
        "type": "group_completed",
        "data": {
            "group_id": group_id
        }
    }
    redis_conn.xadd(STREAMS["pool_to_generator"], {"data": json.dumps(msg)})
    logging.info(f"[{os.getpid()}](QUE) Group completion message sent: {group_id}")


def run(config: dict):
    """
    Main running function of Question Pool Service.
    Responsible for maintaining question pool, updating question dependencies and status, responding to add question and complete question requests.
    """
    # Setup logging
    parent_dir = config.get("output_parent_dir", "logs")
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
    
    # Initialize Updater instance
    updater = Updater(config)
    
    # Connect to Redis
    redis_conn = get_redis_connection(config)
    
    # Setup consumer group, only need to listen to one request stream
    pool_requests_stream = STREAMS["pool_requests"]
    pool_responses_stream = STREAMS["pool_responses"]
    
    # Create consumer group
    try:
        redis_conn.xgroup_create(pool_requests_stream, "pool_group", id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](QUE) Pool requests consumer group already exists: {e}")
        pass
    
    logging.info(f"[{os.getpid()}](QUE) Question Pool service started.")
    
    while True:
        try:
            # Listen to single request stream
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
                        
                        logging.info(f"[{os.getpid()}](QUE) Received request {request_id}, type: {request_type}, sender: {sender}")
                        
                        # Dispatch processing based on request type
                        if request_type == "add_question":
                            # Process add question request
                            try:
                                updater.add_question(request_data)
                                logging.info(f"[{os.getpid()}](QUE) Question {request_data['id']} added successfully")
                                
                                # Send success response
                                response = {
                                    "request_id": request_id,
                                    "status": "success",
                                    "type": "question_added",
                                    "data": {"id": request_data['id']}
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                
                            except Exception as e:
                                logging.exception(f"[{os.getpid()}](QUE) Error adding question: {e}")
                                # Send error response
                                response = {
                                    "request_id": request_id,
                                    "status": "error",
                                    "type": "question_added",
                                    "error": str(e)
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        elif request_type == "complete_question":
                            # Process complete question request
                            try:
                                updater.complete_question(request_data)
                                logging.info(f"[{os.getpid()}](QUE) Question {request_data['id']} marked as completed")
                                
                                # Send success response
                                response = {
                                    "request_id": request_id,
                                    "status": "success", 
                                    "type": "question_completed",
                                    "data": {"id": request_data['id']}
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                
                            except Exception as e:
                                logging.error(f"[{os.getpid()}](QUE) Error completing question: {e}")
                                # Send error response
                                response = {
                                    "request_id": request_id,
                                    "status": "error",
                                    "type": "question_completed",
                                    "error": str(e)
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        elif request_type == "select_question":
                            # Select a question
                            selected_question = updater.select_question()
                            
                            if selected_question:
                                # Set start processing time
                                if "time" not in selected_question:
                                    selected_question["time"] = {}
                                selected_question["time"]["start"] = time.time()
                                
                                # Return question
                                response = {
                                    "request_id": request_id,
                                    "status": "success",
                                    "type": "question_selected",
                                    "data": selected_question
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                logging.info(f"[{os.getpid()}](QUE) Sent question {selected_question['id']} to requester")
                            
                            else:
                                # No available questions
                                response = {
                                    "request_id": request_id,
                                    "status": "empty",
                                    "type": "question_selected",
                                    "data": None
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        elif request_type == "clear_buffer":
                            # Clear internal buffer
                            try:
                                updater.clear_buffer()
                                logging.info(f"[{os.getpid()}](QUE) Question buffer cleared successfully.")
                                response = {
                                    "request_id": request_id,
                                    "status": "success",
                                    "type": "buffer_cleared"
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                            except Exception as e:
                                logging.error(f"[{os.getpid()}](QUE) Error clearing buffer: {e}")
                                response = {
                                    "request_id": request_id,
                                    "status": "error",
                                    "type": "buffer_cleared",
                                    "error": str(e)
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        elif request_type == "answer_question":
                            # Process add answer request
                            try:
                                question_id = request_data.get('id')
                                
                                try:
                                    # Check if the question already exists in buffer
                                    existing_question = updater.get_question_by_id(question_id)
                                    
                                    if existing_question["status"] == "completed":
                                        # Update question answer and status
                                        updater.answer_question(request_data)
                                        logging.info(f"[{os.getpid()}](QUE) Answer added to question {question_id}")

                                        # Check if question group is fully completed
                                        group_id = get_current_group_id(redis_conn)
                                        if group_id and updater.is_group_completed(redis_conn, group_id):
                                            send_group_completed(redis_conn, group_id, request_id)
                                        
                                        # Send success response
                                        response = {
                                            "request_id": request_id,
                                            "status": "success",
                                            "type": "answer_added",
                                            "data": {"id": question_id}
                                        }
                                        redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                    
                                    else:
                                        # Status is not completed
                                        msg = f"Question {question_id} status is {existing_question['status']}, not 'completed'"
                                        logging.error(f"[{os.getpid()}](QUE) {msg}")
                                        
                                        response = {
                                            "request_id": request_id,
                                            "status": "error",
                                            "type": "answer_added",
                                            "error": msg
                                        }
                                        redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                                        
                                except ValueError:
                                    # Question not in buffer, indicating question directly enters answering module from finishing module
                                    logging.info(f"[{os.getpid()}](QUE) New answered question {question_id} is not in the buffer. Must be sent to ANS by FIN. Adding to pool.")
                                    
                                    # Directly add the answered question to buffer
                                    updater.add_answered_question_directly(request_data)
                                    logging.info(f"[{os.getpid()}](QUE) Added answered question {question_id} directly to the pool.")
                                    
                                    # Check if question group is fully completed
                                    group_id = get_current_group_id(redis_conn)
                                    if group_id and updater.is_group_completed(redis_conn, group_id):
                                        send_group_completed(redis_conn, group_id, request_id)
                                    
                                    # Send success response
                                    response = {
                                        "request_id": request_id,
                                        "status": "success",
                                        "type": "answer_added",
                                        "data": {"id": question_id}
                                    }
                                    redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                            
                            except Exception as e:
                                logging.error(f"[{os.getpid()}](QUE) Error processing answer: {e}")
                                # Send error response
                                response = {
                                    "request_id": request_id,
                                    "status": "error",
                                    "type": "no_such_question",
                                    "error": str(e)
                                }
                                redis_conn.xadd(pool_responses_stream, {"data": json.dumps(response)})
                        
                        else:
                            # Unknown request type
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
                        # Acknowledge message as processed regardless
                        redis_conn.xack(pool_requests_stream, "pool_group", msg_id)
            
        except Exception as e:
            logging.error(f"[{os.getpid()}](QUE) Unexpected error in Question Pool service: {e}")
            time.sleep(5)  # Brief sleep after unexpected error before continuing
