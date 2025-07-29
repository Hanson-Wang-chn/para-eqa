import os
import json
import time
import logging

from common.redis_client import get_redis_connection, STREAMS, STATS_KEYS
from utils.knowledgebase import KnowledgeBase
from utils.image_processor import decode_image, encode_image


def process_search_request(kb, request_data):
    """处理搜索请求"""
    text = request_data.get('text', '')
    image_data = request_data.get('image_data')
    top_k = int(request_data.get('top_k', 5))
    
    # 解码图像
    image = decode_image(image_data) if image_data else None
    
    # 执行搜索
    try:
        results = kb.search(text, image, top_k)
        
        # 将搜索结果处理为可序列化的格式
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
        logging.error(f"搜索操作失败: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def process_update_request(kb, request_data):
    """处理更新请求"""
    text = request_data.get('text', '')
    image_data = request_data.get('image_data')
    
    # 解码图像
    image = decode_image(image_data) if image_data else None
    
    # 执行更新
    try:
        kb.update_memory(text, image)
        return {
            "status": "success"
        }
    except Exception as e:
        logging.error(f"更新操作失败: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def run(config: dict):
    """
    Memory Service 的主运行函数。
    负责初始化知识库并处理来自其他模块的记忆检索和更新请求。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "results")
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
    
    # 初始化知识库
    kb = KnowledgeBase(config)
    device = config.get("memory", {}).get("device", "cuda")
    logging.info(f"[{os.getpid()}] 知识库初始化完成，使用设备: {device}")
    
    # Redis初始化
    redis_conn = get_redis_connection(config)
    requests_stream = STREAMS["memory_requests"]
    responses_stream = STREAMS["memory_responses"]
    group_name = "memory_group"
    
    # 尝试创建消费者组，如果已存在则会报错，但可以安全地忽略
    try:
        redis_conn.xgroup_create(requests_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Memory group '{group_name}' already exists. Continuing...")
    
    logging.info(f"[{os.getpid()}] Memory service started. Waiting for requests...")
    
    # 初始化统计计数器
    search_count = 0
    update_count = 0
    
    while True:
        try:
            # 阻塞式地从请求流中读取消息
            messages = redis_conn.xreadgroup(group_name, "memory_worker", {requests_stream: '>'}, count=1, block=None)
            
            if not messages:
                time.sleep(0.01)
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    request_data = json.loads(data['data'])
                    request_id = request_data.get('id')
                    operation = request_data.get('operation')
                    
                    logging.info(f"[{os.getpid()}] 收到请求 {request_id}, 操作类型: {operation}")
                    
                    # 处理请求
                    if operation == "search":
                        result = process_search_request(kb, request_data)
                        search_count += 1
                    elif operation == "update":
                        result = process_update_request(kb, request_data)
                        update_count += 1
                    else:
                        result = {
                            "status": "error",
                            "error": f"不支持的操作: {operation}"
                        }
                    
                    # 发送响应
                    response = {
                        "request_id": request_id,
                        **result
                    }
                    redis_conn.xadd(responses_stream, {"data": json.dumps(response)})
                    
                    # 更新统计信息
                    pipe = redis_conn.pipeline()
                    pipe.hset(STATS_KEYS["memory"], "search_requests", search_count)
                    pipe.hset(STATS_KEYS["memory"], "update_requests", update_count)
                    pipe.hset(STATS_KEYS["memory"], "total_requests", search_count + update_count)
                    pipe.execute()
                    
                    logging.info(f"[{os.getpid()}] 请求 {request_id} 处理完成，状态: {result['status']}")
                    
                    # 确认消息处理完毕
                    redis_conn.xack(requests_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"[{os.getpid()}] Memory service发生错误: {e}")
            time.sleep(5)  # 发生错误时等待一段时间再重试


"""
API Design:

Search Request:
{
    "id": "unique-request-id(uuid)",
    "operation": "search",
    "text": "查询文本描述",
    "image_data": "base64编码的图像数据或null",
    "top_k": 5
}

Update Request:
{
    "id": "unique-request-id(uuid)", 
    "operation": "update",
    "text": "要添加的文本描述",
    "image_data": "base64编码的图像数据或null"
}

Successful Response:
{
    "request_id": "原请求ID",
    "status": "success",
    "data": [
        {
            "id": "memory-item-id",
            "text": "记忆文本",
            "image_data": "base64编码图像或null"
        }
    ]
}

Error Response:
{
    "request_id": "原请求ID", 
    "status": "error",
    "error": "错误描述"
}

"""
