# common/redis_client.py

import redis


# ----------------- Key 前缀与名称设计 -----------------

# 用于存储 Question Pool 中问题元数据的 Hash Key 前缀
KEY_PREFIXES = {
    "question": "question:"
}

# 统计信息
STATS_KEYS = {
    "parser":   "stats:parser",
    "finishing": "stats:finishing",
    "stopping": "stats:stopping",
    "answering": "stats:answering",
    "planner":  "stats:planner",
    "memory": "stats:memory"
}

# 用于存储一组问题的元信息
GROUP_INFO = {
    "group_id": "group_id:",
    "num_questions_init": "num_questions_init:",
    "num_questions_follow_up": "num_questions_follow_up:",
    "correct_answers": "correct_answers:",
    "scene": "scene:",
    "floor": "floor:",
    "max_steps": "max_steps:",
    "angle": "angle:",
    "pts": "pts:", # 包含x、y、z坐标
    "rotation": "rotation:",
    "floor_height": "floor_height:",
    "scene_size": "scene_size:"
}

# # 用于记录每一组问题的运行结果，包括组信息（总运行时间、平均步数等）和每个问题的结果（步数、是否正确等）
# RESULTS_KEYS = {
#     "group_results": "group_results:",  # 每组问题的结果
# }

# 用于持久化任务队列的 Stream 名称
STREAMS = {
    "new_questions": "stream:new_questions", # New Questions -> Parser
    "parser_to_finishing": "stream:parser_to_finishing", # Parser -> Finishing Module
    # "finishing_to_pool": "stream:finishing_to_pool", # Finishing Module -> Question Pool
    # "stopping_to_pool": "stream:stopping_to_pool", # Stopping Module -> Question Pool
    "stopping_to_planner": "stream:stopping_to_planner", # Stopping Module -> Planner
    # "planner_to_pool": "stream:planner_to_pool", # Planner -> Question Pool
    "planner_to_stopping": "stream:planner_to_stopping", # Planner -> Stopping Module
    # "answering_to_pool": "stream:answering_to_pool", # Answering Module -> Question Pool
    # "pool_to_planner": "stream:pool_to_planner", # Question Pool -> Planner
    "to_answering": "stream:to_answering", # Finishing/Stopping Module -> Answering
    "memory_requests": "stream:memory_requests", # Others -> Memory Module
    "memory_responses": "stream:memory_responses", # Memory Module -> Others
    "generator_to_memory": "stream:generator_to_memory", # Generator -> Memory Module
    # "pool_to_generator": "stream:pool_to_generator", # Question Pool -> Generator
    "pool_requests": "stream:pool_requests", # Others -> Question Pool
    "pool_responses": "stream:pool_responses" # Question Pool -> Others
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
  