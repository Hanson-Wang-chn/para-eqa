# services/selector_service.py

"""
# 在selector_service.py中订阅question_pool_service.py的实现示例
def run(config: dict):
    # ...初始化代码...
    
    # 设置消费者组
    selector_stream = STREAMS["pool_to_selector"]
    try:
        redis_conn.xgroup_create(selector_stream, "selector_group", id='0', mkstream=True)
    except Exception as e:
        logging.info(f"Selector consumer group already exists: {e}")
    
    while True:
        try:
            # 从stream读取更新，设置短超时以避免长时间阻塞
            messages = redis_conn.xreadgroup(
                "selector_group", "selector_worker",
                {selector_stream: '>'}, 
                count=1, block=100  # 0.1秒超时
            )
            
            if messages:
                for _, msg_list in messages:
                    for msg_id, data in msg_list:
                        # 处理消息
                        buffer_data = json.loads(data.get('data', '{}'))
                        process_buffer_update(buffer_data)
                        
                        # 确认消息
                        redis_conn.xack(selector_stream, "selector_group", msg_id)
            
            # 其他逻辑...
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Selector service error: {e}")
            time.sleep(1)
"""
