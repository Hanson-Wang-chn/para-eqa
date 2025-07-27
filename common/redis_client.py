# common/redis_client.py

import redis


# ----------------- Key 前缀与名称设计 -----------------

# 用于存储 Question Pool 中问题元数据的 Hash Key 前缀
KEY_PREFIXES = {
    "question": "question:"
}

# 用于分布式锁的 Key
LOCK_KEYS = {
    "selector": "lock:selector"
}

# 统计信息
STATS_KEYS = {
    "parser":   "stats:parser",
    "finishing": "stats:finishing",
    "answering": "stats:answering",
    "planner":  "stats:planner",
    # ...以后有更多service可以继续加
}

# 用于持久化任务队列的 Stream 名称
STREAMS = {
    "new_questions": "stream:new_questions", # New Questions -> Parser
    "parsed_questions": "stream:parsed_questions", # Parsing -> Finishing Module
    "to_answering": "stream:to_answering" # Finishing Module -> Answering
}

# 用于瞬时“唤醒”信号的 Pub/Sub 频道名称
PUBSUB_CHANNELS = {
    "pool_changed": "channel:pool_changed",
    "planner_idle": "channel:planner_idle"
}


# ----------------- 连接函数 -----------------

def get_redis_connection(config: dict) -> redis.Redis:
    """
    根据配置文件创建并返回一个Redis连接实例。
    """
    redis_config = config.get('redis', {})
    pool = redis.ConnectionPool(
        host=redis_config.get('host', 'localhost'),
        port=redis_config.get('port', 6379),
        db=redis_config.get('db', 0),
        decode_responses=True
    )
    return redis.Redis(connection_pool=pool)
  