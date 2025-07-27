# services/finishing_service.py

import os
import json
import time
from common.redis_client import get_redis_connection, STREAMS, PUBSUB_CHANNELS, KEY_PREFIXES

# 伪函数，代表与 Memory 和 VLM 的交互
def check_confidence_with_memory(question_parsed: dict, memory_manager) -> float:
    print(f"[{os.getpid()}] FINISHER: Checking confidence for question {question_parsed['id']}...")
    # m = memory_manager.retrieve(question_parsed)
    # confidence = G_V(prompt, q, m)
    # 模拟一个决策过程
    if "local" in question_parsed.get('scope_type', ''):
        return 0.9 # 假设本地问题可以直接回答
    return 0.4 # 假设全局问题需要探索

def run(config: dict):
    """Finishing Service 的主运行函数"""
    redis_conn = get_redis_connection(config)
    # memory_manager = MemoryManager(config) # 初始化记忆管理器
    
    # 设置消费者组，用于接收来自Parser的任务
    stream_name = STREAMS["parsed_questions"]
    group_name = "finishing_group"
    try:
        redis_conn.xgroup_create(stream_name, group_name, id='$', mkstream=True)
    except Exception:
        pass # 组已存在

    print(f"[{os.getpid()}] Finishing service started.")

    while True:
        try:
            # 1. 优先处理来自 Parser 的新问题
            # 使用 block=1000 (1秒) 而不是无限阻塞，以便有机会检查池中自信的问题
            messages = redis_conn.xreadgroup(group_name, "finisher_worker", {stream_name: '>'}, count=1, block=1000)
            
            if messages:
                for stream, msg_list in messages:
                    for msg_id, data in msg_list:
                        q_parsed = json.loads(data['data'])
                        confidence = check_confidence_with_memory(q_parsed, None)

                        if confidence > config.get('confidence_threshold', 0.8):
                            # 直接发送给 Answering Module
                            print(f"[{os.getpid()}] FINISHER: High confidence. Sending {q_parsed['id']} to Answering Module.")
                            redis_conn.xadd(STREAMS['to_answering'], {'data': json.dumps(q_parsed)})
                        else:
                            # **【Redis Hashes 核心代码】**
                            # 添加到 Question Pool (即写入Redis Hash)
                            q_id = q_parsed['id']
                            question_metadata = {
                                "id": q_id,
                                "description": q_parsed.get('description', ''),
                                "urgency": q_parsed.get('urgency', 0.5),
                                "scope_type": q_parsed.get('scope_type', 'global'),
                                # 初始状态为 "ready"，可以被 Selector 选择
                                "status": "ready", 
                                "cost_estimate": -1.0, # 待Updater更新
                                "reward_estimate": -1.0, # 待Updater更新
                                "dependency": json.dumps(q_parsed.get('dependency', [])),
                            }
                            
                            # 使用 HSET 一次性写入所有字段
                            redis_conn.hset(f"{KEY_PREFIXES['question']}{q_id}", mapping=question_metadata)
                            print(f"[{os.getpid()}] FINISHER: Low confidence. Added {q_id} to Question Pool.")

                            # **【Redis Pub/Sub 核心代码】**
                            # 发布信号，通知 Updater 池已改变
                            redis_conn.publish(PUBSUB_CHANNELS['pool_changed'], q_id)

                        redis_conn.xack(stream_name, group_name, msg_id)
                continue # 处理完新问题后，重新开始循环

            # 2. 如果没有新问题，检查池中是否有被 Stopping Module 标记为 "confident" 的问题
            # (此部分逻辑为轮询，可以根据实际需求优化)
            question_keys = redis_conn.keys(f"{KEY_PREFIXES['question']}*")
            for key in question_keys:
                status = redis_conn.hget(key, "status")
                if status == "confident":
                    # 获取问题数据，发送给Answering，并更新状态
                    q_data = redis_conn.hgetall(key)
                    print(f"[{os.getpid()}] FINISHER: Found confident question {q_data['id']}. Sending to Answering.")
                    redis_conn.xadd(STREAMS['to_answering'], {'data': json.dumps(q_data)})
                    # 更新状态为 "completed"，防止重复发送
                    redis_conn.hset(key, "status", "completed")
                    # 再次通知池已改变
                    redis_conn.publish(PUBSUB_CHANNELS['pool_changed'], q_data['id'])
                    break # 一次只处理一个

        except Exception as e:
            print(f"[{os.getpid()}] ERROR in Finishing service: {e}")
            time.sleep(5)
