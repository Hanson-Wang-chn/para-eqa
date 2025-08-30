# run_para_eqa.py

"""
Before `python run_para_eqa.py`, run `docker run -d --name para-eqa-redis -p 6379:6379 redis:latest` or `docker start para-eqa-redis` to start Redis.
"""

"""
1. Read configuration file
2. Start and initialize each microservice
3. Graceful shutdown of each microservice
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
    """Delete the log directory if it exists"""
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
        print(f"Deleted result directory: {result_dir}")
    else:
        print(f"Result directory does not exist, no need to delete: {result_dir}")


def load_config(config_file):
    """Load YAML configuration file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def clear_redis(config):
    """Clear Redis"""
    redis_conn = get_redis_connection(config)
    logging.info("Flushing Redis DB...")
    redis_conn.flushdb()
    logging.info("Redis DB flushed successfully.")


def listen_for_shutdown(config, processes, shutdown_event):
    """
    Listen for system shutdown requests and trigger shutdown event
    
    Args:
        config: Configuration dictionary
        processes: List of service processes
        shutdown_event: Shutdown event for notifying main thread
    """
    # Connect to Redis
    redis_conn = get_redis_connection(config)
    system_shutdown_stream = STREAMS["system_shutdown"]
    
    # Create consumer group
    group_name = "main_shutdown_group"
    try:
        redis_conn.xgroup_create(system_shutdown_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # Group may already exist, ignore error
        pass
    
    logging.info("Main process started listening for system shutdown requests...")
    
    while not shutdown_event.is_set():
        try:
            # Read messages from stream
            messages = redis_conn.xreadgroup(
                group_name, "main_worker", 
                {system_shutdown_stream: '>'}, 
                count=1, block=1000  # 1 second timeout
            )
            
            if not messages:
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    try:
                        message = json.loads(data.get('data', '{}'))
                        logging.info(f"Received system shutdown request: {message}")
                        
                        # Acknowledge message as processed
                        redis_conn.xack(system_shutdown_stream, group_name, message_id)
                        
                        # Set shutdown event to notify main thread
                        shutdown_event.set()
                        return
                    
                    except Exception as e:
                        logging.error(f"Error processing system shutdown request: {e}")
                        # Acknowledge message to prevent reprocessing error messages
                        redis_conn.xack(system_shutdown_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"Error listening for system shutdown requests: {e}")
            time.sleep(1)


def shutdown_services(processes):
    """
    Gracefully shutdown all services
    
    Args:
        processes: List of service processes
    """
    for p in processes:
        p.terminate()
        p.join()
    logging.info("All services have been shut down gracefully.")


if __name__ == "__main__":
    # Set multiprocessing start method
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
    
    # Clear Redis
    clear_redis(config)

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
    
    # Create shutdown event
    shutdown_event = threading.Event()
    
    # Start listening thread
    shutdown_thread = threading.Thread(target=listen_for_shutdown, args=(config, processes, shutdown_event))
    shutdown_thread.daemon = True  # Set as daemon thread, automatically ends when main thread ends
    shutdown_thread.start()
    
    try:
        while True:
            # If shutdown event is triggered, gracefully shutdown services
            if shutdown_event.is_set():
                logging.info("\nShutdown signal received. Terminating all services...")
                clear_redis(config)
                shutdown_services(processes)
                break
            time.sleep(1)
    
    except KeyboardInterrupt:
        logging.info("\nKeyboardInterrupt received. Terminating all services...")
        shutdown_event.set()  # Set event so listening thread can also exit
        time.sleep(1)
        shutdown_services(processes)
        clear_redis(config)
