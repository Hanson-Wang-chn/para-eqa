# common/redis_client.py

import redis


# ----------------- Key Prefix and Name Design -----------------

# Statistics information
STATS_KEYS = {
    "parser":   "stats:parser",
    "finishing": "stats:finishing",
    "stopping": "stats:stopping",
    "answering": "stats:answering",
    "planner":  "stats:planner",
    "memory": "stats:memory"
}


# Separately record the group ID of the current question
CURRENT_GROUP_ID = "current_group_id"


# Used to store metadata for a group of questions
GROUP_INFO = {
    "group_id": "group_id:",
    "num_questions_init": "num_questions_init:",
    "num_questions_follow_up": "num_questions_follow_up:",
    "correct_answers": "correct_answers:",
    "scene": "scene:",
    "floor": "floor:",
    "max_steps": "max_steps:",
    "angle": "angle:",
    "pts": "pts:", # Contains x, y, z coordinates
    "rotation": "rotation:",
    "floor_height": "floor_height:",
    "scene_size": "scene_size:"
}


# Stream names for persistent task queues
STREAMS = {
    "generator_to_parser": "stream:generator_to_parser", # Generator -> Parser
    "parser_to_finishing": "stream:parser_to_finishing", # Parser -> Finishing Module
    "stopping_to_planner": "stream:stopping_to_planner", # Stopping Module -> Planner
    "planner_to_stopping": "stream:planner_to_stopping", # Planner -> Stopping Module
    "to_answering": "stream:to_answering", # Finishing/Stopping Module -> Answering
    "memory_requests": "stream:memory_requests", # Others -> Memory Module
    "memory_responses": "stream:memory_responses", # Memory Module -> Others
    "generator_to_memory": "stream:generator_to_memory", # Generator -> Memory Module
    "pool_requests": "stream:pool_requests", # Others -> Question Pool
    "pool_responses": "stream:pool_responses", # Question Pool -> Others
    "pool_to_generator": "stream:pool_to_generator", # Question Pool -> Generator
    "system_shutdown": "stream:system_shutdown" # Generator -> run_para_eqa.py
}


# ----------------- Connection Function -----------------

def get_redis_connection(config: dict) -> redis.Redis:
    """
    Create and return a Redis connection instance based on the configuration file.
    """
    redis_config = config.get('redis', {})
    pool = redis.ConnectionPool(
        host=redis_config.get('host', 'localhost'),
        port=redis_config.get('port', 6379),
        db=redis_config.get('db', 0),
        decode_responses=True
    )
    return redis.Redis(connection_pool=pool)
