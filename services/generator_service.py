# service/generator_service.py

# TODO: 接受来自question_pool_service.py的组完成消息后，进行下一组问题的发送或终止程序

import os
import json
import time
import logging
import uuid
import yaml
import glob

from common.redis_client import get_redis_connection, STREAMS, GROUP_INFO


def load_question_data(file_path):
    """
    从YAML文件中加载问题数据
    
    Args:
        file_path: YAML文件路径
        
    Returns:
        dict: 包含问题组信息的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading question data from {file_path}: {e}")
        return None


def store_group_info(redis_conn, group_data):
    """
    将组信息存储到Redis中
    
    Args:
        redis_conn: Redis连接对象
        group_data: 组数据字典
        
    Returns:
        dict: 包含问题ID和答案的映射以及准备好的问题列表
    """
    group_id = group_data.get('group_id') or ''
    scene = group_data.get('scene') or ''
    init_x = group_data.get('init_x') or 0.0
    init_y = group_data.get('init_y') or 0.0
    init_z = group_data.get('init_z') or 0.0
    init_angle = group_data.get('init_angle') or 0.0
    floor = group_data.get('floor', 0)
    
    # 计算问题数量
    questions_init = group_data.get('questions_init', [])
    questions_follow_up = group_data.get('questions_follow_up', [])
    num_questions_init = len(questions_init)
    num_questions_follow_up = len(questions_follow_up)
    
    # 为每个问题生成ID并映射答案
    question_ids_to_answers = {}
    processed_init_questions = []
    processed_follow_up_questions = []
    
    # 处理初始问题
    for q in questions_init:
        q_id = str(uuid.uuid4())
        answer = q.get('answer')
        question_ids_to_answers[q_id] = answer if answer is not None else ''
        processed_init_questions.append({
            "id": q_id,
            "description": q.get('question', '') or ''
        })
    
    # 处理后续问题
    for q in questions_follow_up:
        q_id = str(uuid.uuid4())
        answer = q.get('answer')
        question_ids_to_answers[q_id] = answer if answer is not None else ''
        processed_follow_up_questions.append({
            "id": q_id,
            "description": q.get('question', '') or ''
        })
    
    # 将信息存入Redis
    pipe = redis_conn.pipeline()
    
    # 存储基本信息
    pipe.set(f"{GROUP_INFO['group_id']}{group_id}", group_id)
    pipe.set(f"{GROUP_INFO['scene']}{group_id}", scene)
    pipe.set(f"{GROUP_INFO['angle']}{group_id}", init_angle)
    
    # 存储楼层信息
    pipe.set(f"{GROUP_INFO['floor']}{group_id}", floor)
    
    # 存储坐标信息
    pts = {"x": init_x, "y": init_y, "z": init_z}
    pipe.hset(f"{GROUP_INFO['pts']}{group_id}", mapping=pts)
    
    # 存储问题数量
    pipe.set(f"{GROUP_INFO['num_questions_init']}{group_id}", num_questions_init)
    pipe.set(f"{GROUP_INFO['num_questions_follow_up']}{group_id}", num_questions_follow_up)
    
    # 存储答案映射
    pipe.hset(f"{GROUP_INFO['correct_answers']}{group_id}", mapping=question_ids_to_answers)
    
    # 清空当前组中未使用的GROUP_INFO键
    unused_keys = ["floor", "max_steps", "rotation", "floor_height", "scene_size"]
    for key in unused_keys:
        pipe.delete(f"{GROUP_INFO[key]}{group_id}")
    
    # 执行所有Redis命令
    pipe.execute()
    
    logging.info(f"[{os.getpid()}](GEN) 已存储组信息到Redis，组ID: {group_id}，场景: {scene}")
    logging.info(f"[{os.getpid()}](GEN) 初始问题数: {num_questions_init}，后续问题数: {num_questions_follow_up}")
    
    # 返回处理后的问题组
    return {
        'questions_init': processed_init_questions,
        'questions_follow_up': processed_follow_up_questions
    }


def send_init_questions(redis_conn, questions, stream_name):
    """
    发送初始问题组
    
    Args:
        redis_conn: Redis连接对象
        questions: 问题列表
        stream_name: 目标Stream名称
        
    Returns:
        int: 发送的问题数量
    """
    total_sent = 0
    
    for question in questions:
        redis_conn.xadd(stream_name, {"data": json.dumps(question)})
        total_sent += 1
        logging.info(f"[{os.getpid()}](GEN) 已发送初始问题 {total_sent}/{len(questions)}: '{question['description'][:40]}...'")
    
    return total_sent


def send_follow_up_questions(redis_conn, questions, stream_name, interval):
    """
    发送后续问题组，每个问题之间有间隔时间
    
    Args:
        redis_conn: Redis连接对象
        questions: 问题列表
        stream_name: 目标Stream名称
        interval: 问题间隔时间（秒）
        
    Returns:
        int: 发送的问题数量
    """
    total_sent = 0
    
    for i, question in enumerate(questions):
        # 等待指定的间隔时间
        if i > 0:  # 第一个问题不等待
            logging.info(f"[{os.getpid()}](GEN) 等待 {interval} 秒后发送后续问题...")
            time.sleep(interval)
        
        redis_conn.xadd(stream_name, {"data": json.dumps(question)})
        total_sent += 1
        logging.info(f"[{os.getpid()}](GEN) 已发送后续问题 {i+1}/{len(questions)}: '{question['description'][:40]}...'")
    
    return total_sent


def scan_question_files(directory):
    """
    扫描目录获取所有问题文件
    
    Args:
        directory: 问题文件目录
        
    Returns:
        list: 问题文件路径列表
    """
    if not os.path.exists(directory):
        logging.error(f"问题数据目录不存在: {directory}")
        return []
    
    yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
    if not yaml_files:
        logging.warning(f"在 {directory} 中没有找到问题文件 (*.yaml)")
    
    return yaml_files


def wait_for_group_completion(redis_conn, group_id):
    """
    等待Question Pool发送组完成的消息
    
    Args:
        redis_conn: Redis连接对象
        group_id: 当前处理的组ID
        
    Returns:
        bool: 组是否已完成
    """
    logging.info(f"[{os.getpid()}](GEN) 等待组 {group_id} 完成确认...")
    
    # 创建消费者组（如果不存在）
    pool_responses_stream = STREAMS["pool_responses"]
    try:
        redis_conn.xgroup_create(pool_responses_stream, "generator_group", id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](GEN) Generator response group already exists: {e}")
        pass
    
    # 发送检查组完成请求
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "sender": "generator",
        "type": "check_group_completed",
        "data": {"group_id": group_id}
    }
    redis_conn.xadd(STREAMS["pool_requests"], {"data": json.dumps(request)})
    
    # 等待响应
    while True:
        messages = redis_conn.xreadgroup(
            "generator_group", "generator_worker", 
            {pool_responses_stream: '>'}, 
            count=20, block=100
        )
        
        if not messages:
            continue
        
        for _, message_list in messages:
            for message_id, data in message_list:
                try:
                    response = json.loads(data.get('data', '{}'))
                    response_type = response.get('type')
                    
                    # 确认消息已处理
                    redis_conn.xack(pool_responses_stream, "generator_group", message_id)
                    
                    # 检查是否是组完成消息
                    if response_type == "group_completed" and response.get('data', {}).get('group_id') == group_id:
                        logging.info(f"[{os.getpid()}](GEN) 收到组 {group_id} 完成确认")
                        return True
                    
                except Exception as e:
                    logging.warning(f"[{os.getpid()}](GEN) 处理消息时出错: {e}")
                    # 确认消息以防止无限循环
                    redis_conn.xack(pool_responses_stream, "generator_group", message_id)


def clear_memory(redis_conn):
    """
    向Memory Service发送清空知识库的请求
    
    Args:
        redis_conn: Redis连接对象
        
    Returns:
        bool: 操作是否成功
    """
    request_id = str(uuid.uuid4())
    request = {
        "id": request_id,
        "operation": "clear"
    }
    
    memory_requests_stream = STREAMS["memory_requests"]
    redis_conn.xadd(memory_requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](GEN) 已向Memory Service发送清空知识库请求: {request_id}")
    
    return True


def run(config: dict):
    """
    Generator Service 的主运行函数。
    负责按照配置的规则向系统发送问题。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "results")
    logs_dir = os.path.join(parent_dir, "generator_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logging_path = os.path.join(logs_dir, "generator.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    # 读取配置文件
    config_generator = config.get("generator", {})
    question_data_path = config.get("question_data_path", "./data/benchmark")
    interval_seconds = config_generator.get("interval_seconds", 120)
    
    # 连接 Redis
    redis_conn = get_redis_connection(config)
    stream_name = STREAMS["new_questions"]
    
    logging.info(f"[{os.getpid()}](GEN) Generator service started.")
    
    # 查找并处理所有YAML文件
    yaml_files = scan_question_files(question_data_path)
    
    if not yaml_files:
        logging.error(f"[{os.getpid()}](GEN) 未找到问题文件，退出服务")
        return
    
    try:
        # 处理每一个yaml文件
        for file_index, file_path in enumerate(yaml_files):
            logging.info(f"[{os.getpid()}](GEN) 处理问题文件 [{file_index+1}/{len(yaml_files)}]: {file_path}")
            
            # 1. 清空知识库
            clear_memory(redis_conn)
            
            # 2. 加载问题数据
            group_data = load_question_data(file_path)
            if not group_data:
                logging.error(f"[{os.getpid()}](GEN) 无法加载问题数据，跳过此文件")
                continue
            
            group_id = group_data.get('group_id', '')
            
            # 3. 存储组信息到Redis并获取处理后的问题列表
            processed_questions = store_group_info(redis_conn, group_data)
            
            init_questions = processed_questions['questions_init']
            follow_up_questions = processed_questions['questions_follow_up']
            
            logging.info(f"[{os.getpid()}](GEN) 将立即发送 {len(init_questions)} 个初始问题")
            logging.info(f"[{os.getpid()}](GEN) 将每隔 {interval_seconds} 秒发送 {len(follow_up_questions)} 个后续问题")
            
            # 4. 发送初始问题
            init_sent = send_init_questions(redis_conn, init_questions, stream_name)
            
            # 5. 发送后续问题
            follow_up_sent = send_follow_up_questions(redis_conn, follow_up_questions, stream_name, interval_seconds)
            
            logging.info(f"[{os.getpid()}](GEN) 组 {group_id} 的所有问题已发送完毕，共 {init_sent} 个问题")
            
            # logging.info(f"[{os.getpid()}](GEN) 组 {group_id} 的所有问题已发送完毕，共 {init_sent + follow_up_sent} 个问题")
            
            # 6. 等待组完成确认
            if file_index < len(yaml_files) - 1:  # 如果不是最后一个文件，需要等待确认
                wait_for_group_completion(redis_conn, group_id)
        
        logging.info(f"[{os.getpid()}](GEN) 所有问题组处理完毕，总共 {len(yaml_files)} 组")
        
        # 保持进程运行，直到被终止
        while True:
            time.sleep(3600)  # 睡眠1小时，保持进程活跃
            
    except KeyboardInterrupt:
        logging.info(f"[{os.getpid()}](GEN) Generator service received shutdown signal")
    except Exception as e:
        logging.error(f"[{os.getpid()}](GEN) Generator service encountered an error: {e}")
