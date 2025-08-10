# utils/get_current_group_id.py

import logging
import os
from common.redis_client import GROUP_INFO


def get_current_group_id(redis_conn):
    """获取当前活跃的 group_id"""
    keys = redis_conn.keys(f"{GROUP_INFO['group_id']}*")
    if not keys:
        logging.error(f"[{os.getpid()}] 无法找到当前活跃的 group_id")
        return None
    
    # 从键获取 group_id 值
    # TODO: 读取最新的一个键
    group_id_key = keys[-1]
    group_id = redis_conn.get(group_id_key)
    return group_id
