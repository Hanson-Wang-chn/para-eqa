# services/updater_service.py

import os
import time
from common.redis_client import get_redis_connection, PUBSUB_CHANNELS, KEY_PREFIXES

# 伪函数，代表更新 DAG 和估算值的复杂逻辑
def update_dag_and_estimates(redis_conn):
    print(f"[{os.getpid()}] UPDATER: Rebuilding DAG and updating estimates...")
    # 1. 获取池中所有问题
    question_keys = redis_conn.keys(f"{KEY_PREFIXES['question']}*")
    if not question_keys:
        return
        
    pipe = redis_conn.pipeline()
    for key in question_keys:
        pipe.hgetall(key)
    all_questions = pipe.execute()
    
    # 2. 在内存中重建 DAG (伪代码)
    # dag = build_dag(all_questions)
    
    # 3. 重新计算 cost 和 reward (伪代码)
    # new_estimates = calculate_estimates(dag)
    
    # 4. 批量更新 Redis 中的估算值 (伪代码)
    # pipe = redis_conn.pipeline()
    # for q_id, estimates in new_estimates.items():
    #     pipe.hset(f"{KEY_PREFIXES['question']}{q_id}", "cost_estimate", estimates['cost'])
    #     pipe.hset(f"{KEY_KEY_PREFIXES['question']}{q_id}", "reward_estimate", estimates['reward'])
    # pipe.execute()
    print(f"[{os.getpid()}] UPDATER: Update process finished for {len(all_questions)} questions.")

def run(config: dict):
    """Updater Service 的主运行函数"""
    redis_conn = get_redis_connection(config)
    
    # **【Redis Pub/Sub 核心代码】**
    # 订阅 Question Pool 变化的频道
    pubsub = redis_conn.pubsub()
    pubsub.subscribe(PUBSUB_CHANNELS['pool_changed'])

    print(f"[{os.getpid()}] Updater service started. Waiting for pool changes...")

    # 循环监听信号
    for message in pubsub.listen():
        if message['type'] == 'message':
            changed_q_id = message['data']
            print(f"[{os.getpid()}] UPDATER: Received pool_changed signal (due to {changed_q_id}). Triggering update.")
            
            try:
                # 执行核心更新逻辑
                update_dag_and_estimates(redis_conn)
            except Exception as e:
                print(f"[{os.getpid()}] ERROR in Updater logic: {e}")
