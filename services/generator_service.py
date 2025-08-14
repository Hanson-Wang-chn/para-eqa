# service/generator_service.py

import os
import json
import time
import logging
import uuid
import yaml
import glob

from common.redis_client import get_redis_connection, STREAMS, GROUP_INFO, CURRENT_GROUP_ID


def wait_for_group_completion(redis_conn, group_id):
    """
    等待来自Question Pool的组完成消息
    
    Args:
        redis_conn: Redis连接对象
        group_id: 组ID
        
    Returns:
        bool: 是否成功收到组完成消息
    """
    pool_to_generator_stream = STREAMS["pool_to_generator"]
    
    # 创建消费者组
    group_name = "generator_group"
    try:
        redis_conn.xgroup_create(pool_to_generator_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # 组可能已存在，忽略错误
        pass
    
    logging.info(f"[{os.getpid()}](GEN) 等待组 {group_id} 完成消息...")
    
    while True:
        try:
            # 从流中读取消息
            messages = redis_conn.xreadgroup(
                group_name, "generator_worker", 
                {pool_to_generator_stream: '>'}, 
                count=1, block=100
            )
            
            if not messages:
                # logging.info(f"[{os.getpid()}](GEN) 等待组 {group_id} 完成中...")
                time.sleep(0.1)
                continue
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    try:
                        message = json.loads(data.get('data', '{}'))
                        message_type = message.get('type')
                        message_data = message.get('data', {})
                        message_group_id = message_data.get('group_id')
                        
                        logging.info(f"[{os.getpid()}](GEN) 收到消息: {message_type}, 组ID: {message_group_id}")
                        
                        # 确认消息已处理
                        redis_conn.xack(pool_to_generator_stream, group_name, message_id)
                        
                        # 检查消息类型和组ID
                        if message_type == "group_completed" and message_group_id == group_id:
                            logging.info(f"[{os.getpid()}](GEN) 组 {group_id} 已完成处理")
                            return True
                    
                    except Exception as e:
                        logging.error(f"[{os.getpid()}](GEN) 处理组完成消息时出错: {e}")
                        # 确认消息，防止重复处理错误消息
                        redis_conn.xack(pool_to_generator_stream, group_name, message_id)
        
        except Exception as e:
            logging.error(f"[{os.getpid()}](GEN) 等待组完成消息时出错: {e}")
            time.sleep(1)


def send_system_shutdown(redis_conn):
    """
    向主进程发送系统关闭请求
    
    Args:
        redis_conn: Redis连接对象
        
    Returns:
        bool: 是否成功发送关闭请求
    """
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "type": "system_shutdown",
        "data": {
            "timestamp": time.time()
        }
    }
    
    system_shutdown_stream = STREAMS["system_shutdown"]
    redis_conn.xadd(system_shutdown_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](GEN) 已发送系统关闭请求: {request_id}")
    
    return True


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
    num_questions_init = len(questions_init) if questions_init else 0
    num_questions_follow_up = len(questions_follow_up) if questions_follow_up else 0
    if num_questions_init == 0 and num_questions_follow_up == 0:
        raise ValueError(f"问题组 {group_id} 必须包含至少一个初始或后续问题")
    
    # 为每个问题生成ID并映射答案
    question_ids_to_answers = {}
    processed_init_questions = []
    processed_follow_up_questions = []
    
    # 处理初始问题
    if questions_init:
        for q in questions_init:
            q_id = str(uuid.uuid4())
            answer = q.get('answer')
            question_ids_to_answers[q_id] = answer if answer is not None else ''
            processed_init_questions.append({
                "id": q_id,
                "description": q.get('question', '') or '',
                "time": {}
            })
    
    # 处理后续问题
    if questions_follow_up:
        for q in questions_follow_up:
            q_id = str(uuid.uuid4())
            answer = q.get('answer')
            question_ids_to_answers[q_id] = answer if answer is not None else ''
            processed_follow_up_questions.append({
                "id": q_id,
                "description": q.get('question', '') or '',
                "time": {}
            })
    
    # 将信息存入Redis
    pipe = redis_conn.pipeline()
    
    # 设置当前活跃的组ID
    pipe.set(CURRENT_GROUP_ID, group_id)
    
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
        # 添加request时间
        if "time" not in question:
            question["time"] = {}
        question["time"]["request"] = time.time()
        
        redis_conn.xadd(stream_name, {"data": json.dumps(question)})
        total_sent += 1
        logging.info(f"[{os.getpid()}](GEN) 已发送初始问题 {total_sent}/{len(questions)}: '{question['description'][:40]}...'")
    
    return total_sent


def send_follow_up_questions(redis_conn, questions, stream_name, interval):
    """
    发送后续问题组，每个问题之间有间隔时间（包括第一个问题前也等待）
    
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
        # 每个问题前都等待指定的间隔时间
        logging.info(f"[{os.getpid()}](GEN) 等待 {interval} 秒后发送后续问题...")
        time.sleep(interval)
        
        # 添加request时间
        if "time" not in question:
            question["time"] = {}
        question["time"]["request"] = time.time()
        
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


def clear_buffer(redis_conn):
    """
    向Question Pool Service发送清空问题缓冲区的请求
    
    Args:
        redis_conn: Redis连接对象
        
    Returns:
        bool: 操作是否成功
    """
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "sender": "generator",
        "type": "clear_buffer",
        "data": {
            "timestamp": time.time()
        }
    }
    
    pool_requests_stream = STREAMS["pool_requests"]
    redis_conn.xadd(pool_requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](GEN) 已向Question Pool Service发送清空问题缓冲区请求: {request_id}")
    
    return True


def run(config: dict):
    """
    Generator Service 的主运行函数。
    负责按照配置的规则向系统发送问题。
    """
    # 设置日志
    parent_dir = config.get("output_parent_dir", "logs")
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
    enable_follow_up = config_generator.get("enable_follow_up", True)
    
    if enable_follow_up:
        logging.info(f"[{os.getpid()}](GEN) 启用 follow-up 模式，后续问题将每隔 {interval_seconds} 秒发送一个")
    else:
        logging.info(f"[{os.getpid()}](GEN) 禁用 follow-up 模式，所有问题将一次性发送")
    
    # 连接 Redis
    redis_conn = get_redis_connection(config)
    stream_name = STREAMS["generator_to_parser"]
    
    logging.info(f"[{os.getpid()}](GEN) Generator service started.")
    
    # 查找并处理所有YAML文件
    yaml_files = scan_question_files(question_data_path)
    
    # TODO: 在这里指定某一个group
    # yaml_files = sorted(yaml_files)  # 按文件名排序
    yaml_files = sorted(yaml_files)[0:25]
    
    if not yaml_files:
        logging.error(f"[{os.getpid()}](GEN) 未找到问题文件，退出服务")
        send_system_shutdown(redis_conn)
        return
    
    logging.info(f"[{os.getpid()}](GEN) 共 {len(yaml_files)} 个问题组，开始处理")
    
    try:
        # 处理每一个yaml文件
        for file_index, file_path in enumerate(yaml_files):
            is_last_file = file_index == len(yaml_files) - 1
            logging.info(f"[{os.getpid()}](GEN) 处理问题文件 [{file_index+1}/{len(yaml_files)}]: {file_path}")
            
            # 1. 清空知识库
            clear_memory(redis_conn)
            
            # 2. 清空问题缓冲区
            clear_buffer(redis_conn)
            
            # 3. 加载问题数据
            group_data = load_question_data(file_path)
            if not group_data:
                logging.error(f"[{os.getpid()}](GEN) 无法加载问题数据，跳过此文件")
                continue
            
            group_id = group_data.get('group_id', '')
            
            # 4. 存储组信息到Redis并获取处理后的问题列表
            processed_questions = store_group_info(redis_conn, group_data)
            
            init_questions = processed_questions['questions_init']
            follow_up_questions = processed_questions['questions_follow_up']
            
            # 根据enable_follow_up决定发送方式
            if enable_follow_up:
                # 原有逻辑：分别发送初始问题和后续问题
                logging.info(f"[{os.getpid()}](GEN) 将立即发送 {len(init_questions)} 个初始问题")
                logging.info(f"[{os.getpid()}](GEN) 将每隔 {interval_seconds} 秒发送 {len(follow_up_questions)} 个后续问题")
                
                # 5. 发送初始问题
                init_sent = send_init_questions(redis_conn, init_questions, stream_name)
                
                # 6. 发送后续问题
                if follow_up_questions:
                    follow_up_sent = send_follow_up_questions(redis_conn, follow_up_questions, stream_name, interval_seconds)
                    logging.info(f"[{os.getpid()}](GEN) 组 {group_id} 的所有问题已发送完毕，共 {init_sent + follow_up_sent} 个问题")
                
                else:
                    logging.info(f"[{os.getpid()}](GEN) 组 {group_id} 没有后续问题，已发送 {init_sent} 个初始问题")
            
            else:
                # 新逻辑：将所有问题都作为初始问题一次性发送
                all_questions = init_questions + follow_up_questions
                logging.info(f"[{os.getpid()}](GEN) 将一次性发送所有 {len(all_questions)} 个问题")
                
                total_sent = send_init_questions(redis_conn, all_questions, stream_name)
                logging.info(f"[{os.getpid()}](GEN) 组 {group_id} 的所有问题已一次性发送完毕，共 {total_sent} 个问题")
            
            # 7. 等待来自question_pool_service的组完成请求
            wait_for_group_completion(redis_conn, group_id)
            
            # 8. 判断是否是最后一个文件
            if is_last_file:
                logging.info(f"[{os.getpid()}](GEN) 这是最后一个问题组，将发送系统关闭请求")
                send_system_shutdown(redis_conn)
                break  # 退出循环，不再处理其他文件
            else:
                logging.info(f"[{os.getpid()}](GEN) 继续处理下一个问题组")
        
        logging.info(f"[{os.getpid()}](GEN) 所有问题组处理完毕，总共 {len(yaml_files)} 组")
        
        # 保持进程运行，直到被终止
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info(f"[{os.getpid()}](GEN) Generator service received shutdown signal")
    except Exception as e:
        logging.exception(f"[{os.getpid()}](GEN) Generator service encountered an error: {e}")
