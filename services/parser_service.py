# services/parser_service.py

import os
import uuid
import json
import time
import logging
import re

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.vlm_api import VLM_API


def parse_question(desc, model_name="openai/gpt-oss-120b", prompt_parser=None, server="openrouter", base_url=None, api_key=None):
    """
    Parse question description, use large model to extract urgency and scope_type
    
    Args:
        desc (str): Natural language description of the question
        model_api (str): OpenAI model name to use
        
    Returns:
        tuple: (urgency, scope_type)
            - urgency (float): Urgency level, range [0,1]
            - scope_type (str): Scope type, "local" or "global"
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        # Call large model to get response
        vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
        prompt = prompt_parser.replace("{original_question}", desc)
        response = vlm.request_with_retry(image=None, prompt=prompt)[0]
        
        # Extract JSON part
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = response.strip()
        
        # Parse JSON
        try:
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError:
            logging.info(f"Attempt {attempt + 1} failed, returned invalid JSON: {json_str}")
            valid = False
            continue
        
        # Extract data
        urgency = parsed_data.get('urgency')
        scope_type = parsed_data.get('scope_type')
        
        # Validate value ranges
        valid = True
        
        # Check if urgency is a float within [0,1] range
        if not isinstance(urgency, (int, float)) or urgency < 0 or urgency > 1:
            logging.info(f"Warning: urgency value does not comply with rules (should be a numeric value within [0,1] range): {urgency}")
            valid = False
        
        # Check if scope_type is a string from ["local", "global"]
        if scope_type not in ["local", "global"]:
            logging.info(f"Warning: scope_type value does not comply with rules (should be 'local' or 'global'): {scope_type}")
            valid = False
        
        # If validation passes, return result
        if valid:
            return float(urgency), str(scope_type)
        
        # If validation fails, show retry information
        logging.info(f"Attempt {attempt + 1} failed, retrying large model call...")
    
    # Raise error after three failed attempts
    raise ValueError(f"After {max_retries} attempts, still unable to obtain parsing results that comply with rules")


def run(config: dict):
    """
    Main run function for Parser Service.
    Started as an independent process by the main entry script (run_system.py).
    """
    # Set up logging
    parent_dir = config.get("output_parent_dir", "logs")
    logs_dir = os.path.join(parent_dir, "parser_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "parser.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # Set up VLM
    prompt_parser = config.get("prompt", {}).get("parser", None)
    
    config_vlm = config.get("vlm", {}).get("parser", {})
    model_name = config_vlm.get("model", "qwen/qwen2.5-vl-72b-instruct")
    server = config_vlm.get("server", "openrouter")
    base_url = config_vlm.get("base_url", None)
    api_key = config_vlm.get("api_key", None)
    
    # Redis Initialization
    redis_conn = get_redis_connection(config)
    stream_name = STREAMS["generator_to_parser"]
    group_name = "parser_group"

    # Try to create consumer group, will error if already exists, but can be safely ignored
    try:
        redis_conn.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PAR) Parser group '{group_name}' already exists. Continuing...")
        pass

    logging.info(f"[{os.getpid()}](PAR) Parser service started. Waiting for new questions...")
    
    while True:
        try:
            # Blocking read one message from "new questions" stream
            messages = redis_conn.xreadgroup(group_name, "parser_worker", {stream_name: '>'}, count=1, block=None)
            
            if not messages:
                time.sleep(0.01)
                continue

            for stream, message_list in messages:
                for message_id, data in message_list:
                    logging.info(f"[{os.getpid()}](PAR) Parsing message {message_id}...")
                    question_data = json.loads(data['data'])
                    desc = question_data.get("description")
                    q_id = question_data.get("id", str(uuid.uuid4()))
                    
                    if not desc:
                        logging.info(f"[{os.getpid()}](PAR) WARN: Received message without a description. Skipping.")
                        redis_conn.xack(stream_name, group_name, message_id)
                        continue

                    # 1. Create complete Question metadata
                    urgency, scope_type = parse_question(desc, model_name=model_name, prompt_parser=prompt_parser, server=server, base_url=base_url, api_key=api_key)
                    metadata = {
                        "id": q_id,
                        "description": desc,
                        "urgency": urgency,
                        "scope_type": scope_type,
                        "status": "pending",  # Initial status for all questions
                        "cost_estimate": -1.0, # To be updated by Updater
                        "reward_estimate": -1.0, # To be updated by Updater
                        "dependency": [],
                        "answer": "",
                        "max_steps": -1,
                        "used_steps": 0,
                        "time": {
                            "request": question_data.get("time", {}).get("request", 0)
                        }
                    }
                    logging.info(f"[{os.getpid()}](PAR) Question metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
                    
                    # 2. Store metadata in Redis Hash
                    # redis_conn.hset(f"{KEY_PREFIXES['question']}{q_id}", mapping=metadata)
                    
                    # 3. Notify Finishing Module service that there's a new question to process
                    redis_conn.xadd(STREAMS["parser_to_finishing"], {"data": json.dumps(metadata)})
                    
                    # 4. Update statistics
                    redis_conn.hincrby(STATS_KEYS["parser"], "total", 1)
                    
                    logging.info(f"[{os.getpid()}](PAR) Successfully parsed question {q_id}: '{desc[:40]}...'")

                    # 5. Acknowledge message processing completion, remove from pending list
                    redis_conn.xack(stream_name, group_name, message_id)

        except Exception as e:
            logging.info(f"[{os.getpid()}](PAR) An error occurred in Parser service: {e}")
            time.sleep(5) # Wait for a while before retrying when error occurs
