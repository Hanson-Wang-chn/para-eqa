# services/memory_service.py

import os
import json
import time
import logging
import torch

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.knowledgebase import KnowledgeBase
from utils.image_processor import decode_image, encode_image


def process_search_request(kb, request_data):
    """Process search request"""
    text = request_data.get('text', '')
    image_data = request_data.get('image_data')
    top_k = int(request_data.get('top_k', 5))
    
    # Decode image
    image = decode_image(image_data) if image_data else None
    
    # Execute search
    try:
        results = kb.search(text, image, top_k)
        
        # Process search results into serializable format
        serialized_results = []
        for item in results:
            serialized_item = {
                "id": item["id"],
                "text": item["text"],
                "image_data": encode_image(item["image"]) if item["image"] else None
            }
            serialized_results.append(serialized_item)
        
        return {
            "status": "success",
            "data": serialized_results
        }
    except Exception as e:
        logging.error(f"Search operation failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def process_update_request(kb, request_data):
    """Process update request"""
    text = request_data.get('text', '')
    image_data = request_data.get('image_data')
    
    # Decode image
    image = decode_image(image_data) if image_data else None
    
    # Execute update
    try:
        kb.update_memory(text, image)
        return {
            "status": "success"
        }
    except Exception as e:
        logging.error(f"Update operation failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def run(config: dict):
    """
    Main function for Memory Service.
    Responsible for initializing knowledge base and processing memory retrieval and update requests from other modules.
    """
    # Setup logging
    parent_dir = config.get("output_parent_dir", "logs")
    logs_dir = os.path.join(parent_dir, "memory_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "memory.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # Initialize knowledge base
    kb = KnowledgeBase(config)
    device = config.get("memory", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[{os.getpid()}](MEM) Knowledge base initialization completed, using device: {device}")
    
    # Redis initialization
    redis_conn = get_redis_connection(config)
    requests_stream = STREAMS["memory_requests"]
    responses_stream = STREAMS["memory_responses"]
    group_name = "memory_group"
    
    # Try to create consumer group, if it already exists it will raise an error, but can be safely ignored
    try:
        redis_conn.xgroup_create(requests_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](MEM) Memory group '{group_name}' already exists. Continuing...")
        pass
    
    logging.info(f"[{os.getpid()}](MEM) Memory service started. Waiting for requests...")
    
    # Initialize statistics counters
    search_count = 0
    update_count = 0
    
    while True:
        try:
            # Blocking read from request stream
            messages = redis_conn.xreadgroup(group_name, "memory_worker", {requests_stream: '>'}, count=1, block=None)
            
            if not messages:
                time.sleep(0.01)
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    request_data = json.loads(data['data'])
                    request_id = request_data.get('id')
                    operation = request_data.get('operation')
                    
                    logging.info(f"[{os.getpid()}](MEM) Received request {request_id}, operation type: {operation}")

                    # Process request
                    if operation == "search":
                        result = process_search_request(kb, request_data)
                        search_count += 1
                    
                    elif operation == "update":
                        result = process_update_request(kb, request_data)
                        update_count += 1
                    
                    elif operation == "clear":
                        kb.clear()
                        
                        result = {
                            "status": "success",
                            "data": []
                        }
                        
                        logging.info(f"[{os.getpid()}](MEM) KnowledgeBase has been cleared")
                        
                        search_count = 0
                        update_count = 0
                    
                    else:
                        result = {
                            "status": "error",
                            "error": f"Unsupported operation: {operation}"
                        }
                    
                    # Send response
                    response = {
                        "request_id": request_id,
                        **result
                    }
                    redis_conn.xadd(responses_stream, {"data": json.dumps(response)})
                    
                    # Update statistics
                    pipe = redis_conn.pipeline()
                    pipe.hset(STATS_KEYS["memory"], "search_requests", search_count)
                    pipe.hset(STATS_KEYS["memory"], "update_requests", update_count)
                    pipe.hset(STATS_KEYS["memory"], "total_requests", search_count + update_count)
                    pipe.execute()
                    
                    logging.info(f"[{os.getpid()}](MEM) Request {request_id} processing completed, status: {result['status']}")
                    
                    # Acknowledge message processing completion
                    redis_conn.xack(requests_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"[{os.getpid()}](MEM) Memory service error occurred: {e}")
            time.sleep(5)  # Wait for a while before retrying when error occurs


"""
API Design:

Search Request:
{
    "id": "unique-request-id(uuid)",
    "operation": "search",
    "text": "Query text description",
    "image_data": "base64 encoded image data or null",
    "top_k": 5
}

Update Request:
{
    "id": "unique-request-id(uuid)", 
    "operation": "update",
    "text": "Text description to add",
    "image_data": "base64 encoded image data or null"
}

Clear Request:
{
    "id": "unique-request-id(uuid)", 
    "operation": "clear"
}

Successful Response:
{
    "request_id": "Original request ID",
    "status": "success",
    "data": [
        {
            "id": "memory-item-id",
            "text": "Memory text",
            "image_data": "base64 encoded image or null"
        }
    ]
}

Error Response:
{
    "request_id": "Original request ID", 
    "status": "error",
    "error": "Error description"
}

"""
