# run_para_eqa.py

# Before `python run_para_eqa.py`, run `docker run -d --name para-eqa-redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest` to start Redis.

# 1. 读取配置文件
# 2. 启动并初始化各个微服务
# 3. 收集并统计运行结果

import os
import numpy as np
import logging
import csv
import json
import argparse
import tqdm
import yaml
import time
import redis
from multiprocessing import Process, Manager, set_start_method

from common.redis_client import get_redis_connection, STREAMS, KEY_PREFIXES, STATS_KEYS
from services import (
    generator_service,
    parser_service,
    # question_pool_service,
    # memory_service,
    # updater_service, 
    # selector_service, 
    # planner_service, 
    # answering_service, 
    # finishing_service, 
    # stopping_service
)

# np.set_printoptions(precision=3)
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HABITAT_SIM_LOG"] = (
#     "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
# )
# os.environ["MAGNUM_LOG"] = "quiet"


def load_config(config_file):
    """加载YAML配置文件"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def initialize_system(config):
    """清空Redis"""
    redis_conn = get_redis_connection(config['redis'])
    logging.info("Flushing Redis DB...")
    redis_conn.flushdb()


def display_stats(config):
    """从Redis中读取并显示统计信息"""
    redis_conn = get_redis_connection(config['redis'])
    
    # 获取所有统计信息
    all_stats = {}
    for service_name, stats_key in STATS_KEYS.items():
        stats = redis_conn.hgetall(stats_key)
        if stats:
            all_stats[service_name] = stats
    
    # 获取问题状态统计
    question_prefix = KEY_PREFIXES["question"]
    q_keys = redis_conn.keys(f"{question_prefix}*")
    status_counts = {"pending": 0, "ready": 0, "In_progress": 0, "Completed": 0}
    
    if q_keys:
        pipe = redis_conn.pipeline()
        for key in q_keys:
            pipe.hget(key, "status")
        statuses = pipe.execute()
        for status in statuses:
            if status and status in status_counts:
                status_counts[status] += 1

    # 清屏并显示统计信息
    # os.system('cls' if os.name == 'nt' else 'clear')
    print("="*50)
    print("ParaEQA System Monitor")
    print("="*50)
    print(f"Total Questions: {sum(status_counts.values())}")
    print(f"  - Pending:     {status_counts['pending']}")
    print(f"  - Ready:       {status_counts['ready']}")
    print(f"  - In Progress: {status_counts['In_progress']}")
    print(f"  - Completed:   {status_counts['Completed']}")
    print("-"*50)
    
    # 显示各服务的处理统计
    print("Processing Stats:")
    if all_stats:
        for service_name, stats in all_stats.items():
            print(f"  {service_name.title()} Service:")
            for key, value in stats.items():
                print(f"    - {key.replace('_', ' ').title()}: {value}")
    else:
        print("  No processing stats available yet.")
    
    # 显示流队列信息
    print("-"*50)
    print("Stream Queue Status:")
    for stream_name, stream_key in STREAMS.items():
        try:
            stream_info = redis_conn.xinfo_stream(stream_key)
            length = stream_info.get('length', 0)
            print(f"  - {stream_name.replace('_', ' ').title()}: {length} messages")
        except redis.exceptions.ResponseError:
            # 流不存在
            print(f"  - {stream_name.replace('_', ' ').title()}: 0 messages")
    
    print("="*50)
    print("Press Ctrl+C to shut down.")
    
    # 返回是否所有任务都已完成
    total_questions = sum(status_counts.values())
    return total_questions > 0 and status_counts['Completed'] == total_questions


if __name__ == "__main__":
    # 设置多进程启动方式
    set_start_method("spawn", force=True)
    
    # Parse arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-config", "--config_file", default="config/para_eqa.yaml", type=str)
    args = args_parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Set up logging
    parent_dir = config.get("output_parent_dir", "results")
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    logging_path = os.path.join(parent_dir, "main.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # Initialize the system
    initialize_system(config)

    # Run all services
    services_to_run = {
        "Generator": generator_service.run,
        "Parser": parser_service.run,
        # "QuestionPool": question_pool_service.run,
        # "Memory": memory_service.run,
        # "Updater": updater_service.run,
        # "Selector": selector_service.run,
        # "Planner": planner_service.run,
        # "Answerer": answering_service.run,
        # "Finisher": finishing_service.run,
        # "Stopper": stopping_service.run,
    }

    processes = []
    logging.info("\nStarting all services as background processes...")
    for name, target_func in services_to_run.items():
        process = Process(target=target_func, args=(config,))
        process.start()
        processes.append(process)
        logging.info(f"[+] {name} service started (PID: {process.pid})")
    
    try:
        while True:
            all_done = display_stats(config)
            if all_done:
                logging.info("\nAll tasks completed. System is idle.")
            time.sleep(10)
    except KeyboardInterrupt:
        logging.info("\nShutdown signal received. Terminating all services...")
        for p in processes:
            p.terminate()
            p.join()
        logging.info("All services have been shut down gracefully.")




