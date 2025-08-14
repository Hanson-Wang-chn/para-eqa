# run_para_eqa.py

"""
TODO: 总计划
1. 完善 generator 和 NUWL 计算的逻辑，增加问题开始和结束的时间戳（对于follow-up questions，只需要计算结束时间比其开始时间晚的问题的步数；或者直接用时间计算 NUWL）
2. 完善 get_reward_estimate 方法，对于重复的地点给予更高奖励
3. 增加控制字段，对比实验
    - 有无 reward_estimate （在无follow-up questions的条件下验证）
    - 有无在组内各个问题之间分享 memory
4. 数据集
"""

"""
baseline：sequential，无reward，有memory；sequential，无reward，无memory
主实验：有follow-up，有reward，有memory；
reward：无follow-up，有reward，有memory；无follow-up，无reward，有memory
memory：有follow-up，有reward，无memory

控制：use_parallel, enable_reward_estimate, enable_follow_up, use_rag
"""

"""
Before `python run_para_eqa.py`, run `docker run -d --name para-eqa-redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest` or `docker start para-eqa-redis` to start Redis.
"""

"""
1. 读取配置文件
2. 启动并初始化各个微服务
3. 各个微服务的优雅退出
"""

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
import shutil
import threading
from multiprocessing import Process, Manager, set_start_method

from common.redis_client import get_redis_connection, STREAMS
from services import (
    generator_service,
    parser_service,
    question_pool_service,
    memory_service, 
    planner_service, 
    answering_service, 
    finishing_service, 
    stopping_service
)

np.set_printoptions(precision=3)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"


def clear_record(result_dir="logs"):
    """如果日志目录存在，则删除它"""
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
        print(f"已删除结果目录: {result_dir}")
    else:
        print(f"结果目录不存在，无需删除: {result_dir}")


def load_config(config_file):
    """加载YAML配置文件"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def initialize_system(config):
    """清空Redis"""
    redis_conn = get_redis_connection(config['redis'])
    logging.info("Flushing Redis DB...")
    redis_conn.flushdb()


def listen_for_shutdown(config, processes, shutdown_event):
    """
    监听系统关闭请求并触发关闭事件
    
    Args:
        config: 配置字典
        processes: 服务进程列表
        shutdown_event: 关闭事件，用于通知主线程
    """
    # 连接Redis
    redis_conn = get_redis_connection(config['redis'])
    system_shutdown_stream = STREAMS["system_shutdown"]
    
    # 创建消费者组
    group_name = "main_shutdown_group"
    try:
        redis_conn.xgroup_create(system_shutdown_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # 组可能已存在，忽略错误
        pass
    
    logging.info("主进程开始监听系统关闭请求...")
    
    while not shutdown_event.is_set():
        try:
            # 从流中读取消息
            messages = redis_conn.xreadgroup(
                group_name, "main_worker", 
                {system_shutdown_stream: '>'}, 
                count=1, block=1000  # 1秒超时
            )
            
            if not messages:
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    try:
                        message = json.loads(data.get('data', '{}'))
                        logging.info(f"收到系统关闭请求: {message}")
                        
                        # 确认消息已处理
                        redis_conn.xack(system_shutdown_stream, group_name, message_id)
                        
                        # 设置关闭事件，通知主线程
                        shutdown_event.set()
                        return
                    
                    except Exception as e:
                        logging.error(f"处理系统关闭请求时出错: {e}")
                        # 确认消息，防止重复处理错误消息
                        redis_conn.xack(system_shutdown_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"监听系统关闭请求时出错: {e}")
            time.sleep(1)


def shutdown_services(processes):
    """
    优雅地关闭所有服务
    
    Args:
        processes: 服务进程列表
    """
    for p in processes:
        p.terminate()
        p.join()
    logging.info("All services have been shut down gracefully.")


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
    parent_dir = config.get("logs_parent_dir", "logs")
    clear_record(result_dir=parent_dir) # Clear previous records
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
        "QuestionPool": question_pool_service.run,
        "Memory": memory_service.run,
        "Planner": planner_service.run,
        "Answerer": answering_service.run,
        "Finisher": finishing_service.run,
        "Stopper": stopping_service.run,
    }

    processes = []
    logging.info("\nStarting all services as background processes...")
    for name, target_func in services_to_run.items():
        process = Process(target=target_func, args=(config,))
        process.start()
        processes.append(process)
        logging.info(f"[+] {name} service started (PID: {process.pid})")
    
    # 创建关闭事件
    shutdown_event = threading.Event()
    
    # 启动监听线程
    shutdown_thread = threading.Thread(target=listen_for_shutdown, args=(config, processes, shutdown_event))
    shutdown_thread.daemon = True  # 设为守护线程，主线程结束时自动结束
    shutdown_thread.start()
    
    try:
        while True:
            # 如果关闭事件被触发，优雅关闭服务
            if shutdown_event.is_set():
                logging.info("\nShutdown signal received. Terminating all services...")
                shutdown_services(processes)
                break
            time.sleep(1)
    
    except KeyboardInterrupt:
        logging.info("\nKeyboardInterrupt received. Terminating all services...")
        shutdown_event.set()  # 设置事件，让监听线程也能退出
        shutdown_services(processes)
