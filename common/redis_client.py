# common/redis_client.py

import redis


# ----------------- Key 前缀与名称设计 -----------------

# 用于存储 Question Pool 中问题元数据的 Hash Key 前缀
KEY_PREFIXES = {
    "question": "question:"
}

# 用于分布式锁的 Key
"""
updater:
- Question Pool 相关
- Updater 更新 Buffer 和 DAG 前加锁
- Selector 从 Question Pool 选问题时加锁

selector:
- Selector 选问题时加锁，防止多个 Selector 选中同一个问题
- 本项目只考虑一个selector，所以未使用该锁
"""

LOCK_KEYS = {
    "updater": "lock:updater",
    "selector": "lock:selector"
}

# 尝试获取 updater 锁，非阻塞，如果锁被占用则直接跳过
# with redis_conn.lock(LOCK_KEYS['updater'], blocking=False) as lock:
#     if not lock:
#         print(f"[{os.getpid()}] UPDATER: Another update is already in progress. Skipping.")
#         continue
    
#     # --- 成功获取锁，进入临界区 ---
#     try:
#         # 执行核心更新逻辑
#         update_dag_and_estimates(redis_conn)
#     except Exception as e:
#         print(f"[{os.getpid()}] ERROR in Updater logic: {e}")
#     # --- 临界区结束，锁自动释放 ---

# 统计信息
STATS_KEYS = {
    "parser":   "stats:parser",
    "finishing": "stats:finishing",
    "answering": "stats:answering",
    "planner":  "stats:planner",
    "memory": "stats:memory"
}

# 用于持久化任务队列的 Stream 名称
STREAMS = {
    "new_questions": "stream:new_questions", # New Questions -> Parser
    "parsed_questions": "stream:parsed_questions", # Parsing -> Finishing Module
    "to_answering": "stream:to_answering", # Finishing Module -> Answering
    "memory_requests": "stream:memory_requests", # Others -> Memory Module
    "memory_responses": "stream:memory_responses" # Memory Module -> Others
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
  