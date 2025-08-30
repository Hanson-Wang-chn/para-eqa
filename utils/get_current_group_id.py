# utils/get_current_group_id.py

import logging
import os
from common.redis_client import CURRENT_GROUP_ID


def get_current_group_id(redis_conn):
    """Get the current active group_id"""
    group_id = redis_conn.get(CURRENT_GROUP_ID)
    if not group_id:
        logging.error(f"[{os.getpid()}] Unable to find key '{CURRENT_GROUP_ID}' in Redis")
        return None
    
    return group_id
