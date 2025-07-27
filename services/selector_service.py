# services/selector_service.py

import os
import json
import time
from common.redis_client import get_redis_connection, PUBSUB_CHANNELS, LOCK_KEYS, KEY_PREFIXES

# 伪函数，代表选择算法
def select_best_question(ready_questions: list[dict]) -> dict | None:
    if not ready_questions:
        return None
    # 简单的选择策略：选择 urgency 最高的
    return max(ready_questions, key=lambda q: float(q.get('urgency', 0)))

def run(config: dict):
    """Selector Service 的主运行函数"""
    redis_conn = get_redis_connection(config)
    
    # **【Redis Pub/Sub 核心代码】**
    # 订阅 Planner 空闲频道
    pubsub = redis_conn.pubsub()
    pubsub.subscribe(PUBSUB_CHANNELS['planner_idle'])
    
    print(f"[{os.getpid()}] Selector service started. Waiting for planner to be idle...")

    # 循环监听来自 Planner 的信号
    for message in pubsub.listen():
        if message['type'] != 'message':
            continue

        print(f"[{os.getpid()}] SELECTOR: Received planner_idle signal. Attempting to select a question.")

        # **【Redis Locks 核心代码】**
        # 尝试获取全局选择锁，防止多个进程同时选择
        # blocking_timeout=5 表示如果锁被占用，最多等待5秒
        with redis_conn.lock(LOCK_KEYS['selector'], blocking_timeout=5) as lock:
            if not lock:
                print(f"[{os.getpid()}] SELECTOR: Could not acquire lock, another process is selecting.")
                continue

            # --- 成功获取锁，进入临界区 ---
            print(f"[{os.getpid()}] SELECTOR: Lock acquired. Searching for ready questions.")
            
            # 1. 从 Question Pool 中找出所有待处理的问题
            question_keys = redis_conn.keys(f"{KEY_PREFIXES['question']}*")
            ready_questions = []
            if question_keys:
                pipe = redis_conn.pipeline()
                for key in question_keys:
                    pipe.hgetall(key)
                all_questions = pipe.execute()
                
                ready_questions = [q for q in all_questions if q and q.get("status") == "ready"]

            if not ready_questions:
                print(f"[{os.getpid()}] SELECTOR: No ready questions found in the pool.")
                continue # 锁会自动释放

            # 2. 执行选择算法
            selected_q = select_best_question(ready_questions)
            if not selected_q:
                continue

            # 3. 更新选中问题的状态为 "in_progress"
            q_id = selected_q['id']
            redis_conn.hset(f"{KEY_PREFIXES['question']}{q_id}", "status", "in_progress")

            print(f"[{os.getpid()}] SELECTOR: Selected question {q_id}. Sending to Planner.")
            
            # 4. 通知 Planner 开始工作 (可以通过 Pub/Sub 或另一个 Stream)
            # 这里使用 Pub/Sub 简单通知
            redis_conn.publish("channel:to_planner", json.dumps(selected_q))
            # --- 临界区结束，锁自动释放 ---
