# utils/get_current_group_id.py

import logging
import os
from common.redis_client import CURRENT_GROUP_ID


def get_current_group_id(redis_conn):
    """获取当前活跃的 group_id"""
    group_id = redis_conn.get(CURRENT_GROUP_ID)
    if not group_id:
        logging.error(f"[{os.getpid()}] 无法在Redis中找到键 '{CURRENT_GROUP_ID}'")
        return None
    
    return group_id
