# service/generator_service.py

import os
import json
import time
import logging
import uuid
import yaml
import glob

from common.redis_client import get_redis_connection, STREAMS, GROUP_INFO, CURRENT_GROUP_ID


def wait_for_group_completion(redis_conn, group_id):
    """
    Wait for group completion message from Question Pool
    
    Args:
        redis_conn: Redis connection object
        group_id: Group ID
        
    Returns:
        bool: Whether group completion message was successfully received
    """
    pool_to_generator_stream = STREAMS["pool_to_generator"]
    
    # Create consumer group
    group_name = "generator_group"
    try:
        redis_conn.xgroup_create(pool_to_generator_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # Group may already exist, ignore error
        pass
    
    logging.info(f"[{os.getpid()}](GEN) Waiting for group {group_id} completion message...")
    
    while True:
        try:
            # Read messages from stream
            messages = redis_conn.xreadgroup(
                group_name, "generator_worker", 
                {pool_to_generator_stream: '>'}, 
                count=1, block=100
            )
            
            if not messages:
                # logging.info(f"[{os.getpid()}](GEN) Waiting for group {group_id} completion...")
                time.sleep(0.1)
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    try:
                        message = json.loads(data.get('data', '{}'))
                        message_type = message.get('type')
                        message_data = message.get('data', {})
                        message_group_id = message_data.get('group_id')
                        
                        logging.info(f"[{os.getpid()}](GEN) Received message: {message_type}, Group ID: {message_group_id}")
                        
                        # Acknowledge message as processed
                        redis_conn.xack(pool_to_generator_stream, group_name, message_id)
                        
                        # Check message type and group ID
                        if message_type == "group_completed" and message_group_id == group_id:
                            logging.info(f"[{os.getpid()}](GEN) Group {group_id} has completed processing")
                            return True
                    
                    except Exception as e:
                        logging.error(f"[{os.getpid()}](GEN) Error processing group completion message: {e}")
                        # Acknowledge message to prevent reprocessing error messages
                        redis_conn.xack(pool_to_generator_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"[{os.getpid()}](GEN) Error waiting for group completion message: {e}")
            time.sleep(1)


def send_system_shutdown(redis_conn):
    """
    Send system shutdown request to main process
    
    Args:
        redis_conn: Redis connection object
        
    Returns:
        bool: Whether shutdown request was successfully sent
    """
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "type": "system_shutdown",
        "data": {
            "timestamp": time.time()
        }
    }
    
    system_shutdown_stream = STREAMS["system_shutdown"]
    redis_conn.xadd(system_shutdown_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](GEN) System shutdown request sent: {request_id}")
    
    return True


def load_question_data(file_path):
    """
    Load question data from YAML file
    
    Args:
        file_path: YAML file path
        
    Returns:
        dict: Dictionary containing question group information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading question data from {file_path}: {e}")
        return None


def store_group_info(redis_conn, group_data):
    """
    Store group information in Redis
    
    Args:
        redis_conn: Redis connection object
        group_data: Group data dictionary
        
    Returns:
        dict: Dictionary containing question ID to answer mapping and prepared question lists
    """
    group_id = group_data.get('group_id') or ''
    scene = group_data.get('scene') or ''
    init_x = group_data.get('init_x') or 0.0
    init_y = group_data.get('init_y') or 0.0
    init_z = group_data.get('init_z') or 0.0
    init_angle = group_data.get('init_angle') or 0.0
    floor = group_data.get('floor', 0)
    
    # Calculate number of questions
    questions_init = group_data.get('questions_init', [])
    questions_follow_up = group_data.get('questions_follow_up', [])
    num_questions_init = len(questions_init) if questions_init else 0
    num_questions_follow_up = len(questions_follow_up) if questions_follow_up else 0
    if num_questions_init == 0 and num_questions_follow_up == 0:
        raise ValueError(f"Question group {group_id} must contain at least one initial or follow-up question")
    
    # Generate ID for each question and map answers
    question_ids_to_answers = {}
    processed_init_questions = []
    processed_follow_up_questions = []
    
    # Process initial questions
    if questions_init:
        for q in questions_init:
            q_id = str(uuid.uuid4())
            answer = q.get('answer')
            question_ids_to_answers[q_id] = answer if answer is not None else ''
            processed_init_questions.append({
                "id": q_id,
                "description": q.get('question', '') or '',
                "time": {}
            })
    
    # Process follow-up questions
    if questions_follow_up:
        for q in questions_follow_up:
            q_id = str(uuid.uuid4())
            answer = q.get('answer')
            question_ids_to_answers[q_id] = answer if answer is not None else ''
            processed_follow_up_questions.append({
                "id": q_id,
                "description": q.get('question', '') or '',
                "time": {}
            })
    
    # Store information in Redis
    pipe = redis_conn.pipeline()
    
    # Set current active group ID
    pipe.set(CURRENT_GROUP_ID, group_id)
    
    # Store basic information
    pipe.set(f"{GROUP_INFO['group_id']}{group_id}", group_id)
    pipe.set(f"{GROUP_INFO['scene']}{group_id}", scene)
    pipe.set(f"{GROUP_INFO['angle']}{group_id}", init_angle)
    
    # Store floor information
    pipe.set(f"{GROUP_INFO['floor']}{group_id}", floor)
    
    # Store coordinate information
    pts = {"x": init_x, "y": init_y, "z": init_z}
    pipe.hset(f"{GROUP_INFO['pts']}{group_id}", mapping=pts)
    
    # Store number of questions
    pipe.set(f"{GROUP_INFO['num_questions_init']}{group_id}", num_questions_init)
    pipe.set(f"{GROUP_INFO['num_questions_follow_up']}{group_id}", num_questions_follow_up)
    
    # Store answer mapping
    pipe.hset(f"{GROUP_INFO['correct_answers']}{group_id}", mapping=question_ids_to_answers)
    
    # Clear unused GROUP_INFO keys for current group
    unused_keys = ["floor", "max_steps", "rotation", "floor_height", "scene_size"]
    for key in unused_keys:
        pipe.delete(f"{GROUP_INFO[key]}{group_id}")
    
    # Execute all Redis commands
    pipe.execute()
    
    logging.info(f"[{os.getpid()}](GEN) Group information stored in Redis, Group ID: {group_id}, Scene: {scene}")
    logging.info(f"[{os.getpid()}](GEN) Initial questions: {num_questions_init}, Follow-up questions: {num_questions_follow_up}")
    
    # Return processed question groups
    return {
        'questions_init': processed_init_questions,
        'questions_follow_up': processed_follow_up_questions
    }


def send_init_questions(redis_conn, questions, stream_name):
    """
    Send initial question group
    
    Args:
        redis_conn: Redis connection object
        questions: Question list
        stream_name: Target stream name
        
    Returns:
        int: Number of questions sent
    """
    total_sent = 0
    
    for question in questions:
        # Add request time
        if "time" not in question:
            question["time"] = {}
        question["time"]["request"] = time.time()
        
        redis_conn.xadd(stream_name, {"data": json.dumps(question)})
        total_sent += 1
        logging.info(f"[{os.getpid()}](GEN) Initial question sent {total_sent}/{len(questions)}: '{question['description'][:40]}...'")
    
    return total_sent


def send_follow_up_questions(redis_conn, questions, stream_name, interval):
    """
    Send follow-up question group with interval between questions (including waiting before first question)
    
    Args:
        redis_conn: Redis connection object
        questions: Question list
        stream_name: Target stream name
        interval: Interval between questions (seconds)
        
    Returns:
        int: Number of questions sent
    """
    total_sent = 0
    
    for i, question in enumerate(questions):
        # Wait for specified interval before each question
        logging.info(f"[{os.getpid()}](GEN) Waiting {interval} seconds before sending follow-up question...")
        time.sleep(interval)
        
        # Add request time
        if "time" not in question:
            question["time"] = {}
        question["time"]["request"] = time.time()
        
        redis_conn.xadd(stream_name, {"data": json.dumps(question)})
        total_sent += 1
        logging.info(f"[{os.getpid()}](GEN) Follow-up question sent {i+1}/{len(questions)}: '{question['description'][:40]}...'")
    
    return total_sent


def scan_question_files(directory):
    """
    Scan directory for all question files
    
    Args:
        directory: Question file directory
        
    Returns:
        list: List of question file paths
    """
    if not os.path.exists(directory):
        logging.error(f"Question data directory does not exist: {directory}")
        return []
    
    yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
    if not yaml_files:
        logging.warning(f"No question files (*.yaml) found in {directory}")
    
    return yaml_files


def clear_memory(redis_conn):
    """
    Send clear knowledge base request to Memory Service
    
    Args:
        redis_conn: Redis connection object
        
    Returns:
        bool: Whether operation was successful
    """
    request_id = str(uuid.uuid4())
    request = {
        "id": request_id,
        "operation": "clear"
    }
    
    memory_requests_stream = STREAMS["memory_requests"]
    redis_conn.xadd(memory_requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](GEN) Clear knowledge base request sent to Memory Service: {request_id}")
    
    return True


def clear_buffer(redis_conn):
    """
    Send clear question buffer request to Question Pool Service
    
    Args:
        redis_conn: Redis connection object
        
    Returns:
        bool: Whether operation was successful
    """
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "sender": "generator",
        "type": "clear_buffer",
        "data": {
            "timestamp": time.time()
        }
    }
    
    pool_requests_stream = STREAMS["pool_requests"]
    redis_conn.xadd(pool_requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](GEN) Clear question buffer request sent to Question Pool Service: {request_id}")
    
    return True


def run(config: dict):
    """
    Main run function for Generator Service.
    Responsible for sending questions to the system according to configured rules.
    """
    # Setup logging
    parent_dir = config.get("output_parent_dir", "logs")
    logs_dir = os.path.join(parent_dir, "generator_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "generator.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # Read configuration
    config_generator = config.get("generator", {})
    question_data_path = config.get("question_data_path", "./data/benchmark")
    interval_seconds = config_generator.get("interval_seconds", 120)
    enable_follow_up = config_generator.get("enable_follow_up", True)
    start_group = config_generator.get("start_group", None)
    end_group = config_generator.get("end_group", None)
    
    if enable_follow_up:
        logging.info(f"[{os.getpid()}](GEN) Follow-up mode enabled, follow-up questions will be sent every {interval_seconds} seconds")
    else:
        logging.info(f"[{os.getpid()}](GEN) Follow-up mode disabled, all questions will be sent at once")
    
    # Connect to Redis
    redis_conn = get_redis_connection(config)
    stream_name = STREAMS["generator_to_parser"]
    
    logging.info(f"[{os.getpid()}](GEN) Generator service started.")
    
    # Find and process all YAML files
    yaml_files = scan_question_files(question_data_path)
    yaml_files = sorted(yaml_files)  # Sort by filename

    # Select files to process based on start_group and end_group
    total_groups = len(yaml_files)
    if total_groups == 0:
        logging.error(f"[{os.getpid()}](GEN) No question files found, exiting service")
        send_system_shutdown(redis_conn)
        return

    # Process start_group and end_group parameters
    if start_group is None and end_group is None:
        # Process all groups
        selected_files = yaml_files
    elif start_group is None:
        # Start from beginning, end at end_group
        if end_group >= total_groups or end_group < 0:
            logging.warning(f"[{os.getpid()}](GEN) Configured end_group ({end_group}) is out of valid range [0, {total_groups-1}], adjusting to {total_groups-1}")
            end_group = total_groups - 1
        selected_files = yaml_files[:end_group+1]  # +1 because slice doesn't include end index
    elif end_group is None:
        # Start from start_group, go to end
        if start_group >= total_groups or start_group < 0:
            logging.warning(f"[{os.getpid()}](GEN) Configured start_group ({start_group}) is out of valid range [0, {total_groups-1}], adjusting to 0")
            start_group = 0
        selected_files = yaml_files[start_group:]
    else:
        # Both start_group and end_group have values
        if start_group >= total_groups or start_group < 0:
            logging.warning(f"[{os.getpid()}](GEN) Configured start_group ({start_group}) is out of valid range [0, {total_groups-1}], adjusting to 0")
            start_group = 0
        if end_group >= total_groups or end_group < 0:
            logging.warning(f"[{os.getpid()}](GEN) Configured end_group ({end_group}) is out of valid range [0, {total_groups-1}], adjusting to {total_groups-1}")
            end_group = total_groups - 1
        if start_group > end_group:
            logging.warning(f"[{os.getpid()}](GEN) Configured start_group ({start_group}) is greater than end_group ({end_group}), swapping them")
            start_group, end_group = end_group, start_group
        selected_files = yaml_files[start_group:end_group+1]  # +1 because slice doesn't include end index

    yaml_files = selected_files
    logging.info(f"[{os.getpid()}](GEN) Will process {len(yaml_files)} question groups (from index {start_group if start_group is not None else 0} to {end_group if end_group is not None else total_groups-1})")

    if not yaml_files:
        logging.error(f"[{os.getpid()}](GEN) No question files in selected group range, exiting service")
        send_system_shutdown(redis_conn)
        return
    
    logging.info(f"[{os.getpid()}](GEN) Total {len(yaml_files)} question groups, starting processing")
    
    try:
        # Process each yaml file
        for file_index, file_path in enumerate(yaml_files):
            is_last_file = file_index == len(yaml_files) - 1
            logging.info(f"[{os.getpid()}](GEN) Processing question file [{file_index+1}/{len(yaml_files)}]: {file_path}")
            
            # 1. Clear knowledge base
            clear_memory(redis_conn)
            
            # 2. Clear question buffer
            clear_buffer(redis_conn)
            
            # 3. Load question data
            group_data = load_question_data(file_path)
            if not group_data:
                logging.error(f"[{os.getpid()}](GEN) Unable to load question data, skipping this file")
                continue
            
            group_id = group_data.get('group_id', '')
            
            # 4. Store group information in Redis and get processed question lists
            processed_questions = store_group_info(redis_conn, group_data)
            
            init_questions = processed_questions['questions_init']
            follow_up_questions = processed_questions['questions_follow_up']
            
            # Decide sending method based on enable_follow_up
            if enable_follow_up:
                # Original logic: send initial and follow-up questions separately
                logging.info(f"[{os.getpid()}](GEN) Will immediately send {len(init_questions)} initial questions")
                logging.info(f"[{os.getpid()}](GEN) Will send {len(follow_up_questions)} follow-up questions every {interval_seconds} seconds")
                
                # 5. Send initial questions
                init_sent = send_init_questions(redis_conn, init_questions, stream_name)
                
                # 6. Send follow-up questions
                if follow_up_questions:
                    follow_up_sent = send_follow_up_questions(redis_conn, follow_up_questions, stream_name, interval_seconds)
                    logging.info(f"[{os.getpid()}](GEN) All questions for group {group_id} have been sent, total {init_sent + follow_up_sent} questions")
                
                else:
                    logging.info(f"[{os.getpid()}](GEN) Group {group_id} has no follow-up questions, sent {init_sent} initial questions")
            
            else:
                # New logic: send all questions as initial questions at once
                all_questions = init_questions + follow_up_questions
                logging.info(f"[{os.getpid()}](GEN) Will send all {len(all_questions)} questions at once")
                
                total_sent = send_init_questions(redis_conn, all_questions, stream_name)
                logging.info(f"[{os.getpid()}](GEN) All questions for group {group_id} have been sent at once, total {total_sent} questions")
            
            # 7. Wait for group completion request from question_pool_service
            wait_for_group_completion(redis_conn, group_id)
            
            # 8. Check if this is the last file
            if is_last_file:
                logging.info(f"[{os.getpid()}](GEN) This is the last question group, will send system shutdown request")
                send_system_shutdown(redis_conn)
                break  # Exit loop, no more files to process
            else:
                logging.info(f"[{os.getpid()}](GEN) Continue processing next question group")
        
        logging.info(f"[{os.getpid()}](GEN) All question groups processed, total {len(yaml_files)} groups")
        
        # Keep process running until terminated
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info(f"[{os.getpid()}](GEN) Generator service received shutdown signal")
    except Exception as e:
        logging.exception(f"[{os.getpid()}](GEN) Generator service encountered an error: {e}")
