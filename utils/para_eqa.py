# utils/para_eqa.py

import os
import json
import numpy as np
import pickle
import logging
import math
import quaternion
import cv2
from PIL import Image, ImageDraw, ImageFont
import habitat_sim
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import uuid
import time
import base64
from io import BytesIO
import re

from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from utils.habitat import (
    make_simple_config,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from utils.geom import get_cam_intr, get_scene_bnds
from utils.tsdf import TSDFPlanner
from utils.utils import (
    draw_letters,
    save_rgbd,
    display_sample,
    pixel2world
)
from utils.vlm_api import VLM_API
from common.redis_client import get_redis_connection, STREAMS, GROUP_INFO
from utils.image_processor import encode_image
from utils.get_current_group_id import get_current_group_id

np.set_printoptions(precision=3)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"


# =========================================
# ---------- Auxiliary Functions ----------
# =========================================

def search(redis_conn, text, image=None, top_k=5):
    """
    向Memory Service发送搜索请求并等待响应
    
    Args:
        redis_conn: Redis连接对象
        text: 查询文本
        image: PIL图像对象或None
        top_k: 返回的最大结果数量
    
    Returns:
        list: 知识库搜索结果列表
    """
    # 编码图像
    image_data = encode_image(image) if image else None
    
    # 创建请求
    request_id = str(uuid.uuid4())
    request = {
        "id": request_id,
        "operation": "search",
        "text": text,
        "image_data": image_data,
        "top_k": top_k
    }
    
    # 定义流
    requests_stream = STREAMS["memory_requests"]
    responses_stream = STREAMS["memory_responses"]
    
    # 创建消费者组(如果不存在)
    group_name = f"memory_client_{os.getpid()}"
    try:
        redis_conn.xgroup_create(responses_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PLA) Memory response group already exists: {e}")
        pass
    
    # 发送请求
    redis_conn.xadd(requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](PLA) 向Memory发送搜索请求: {request_id}")
    
    # 等待响应
    memory_response = None
    wait_start_time = time.time()
    max_wait_time = 300  # 最长等待时间，单位秒

    while memory_response is None and (time.time() - wait_start_time < max_wait_time):
        try:
            # 使用block参数高效等待
            responses = redis_conn.xreadgroup(
                group_name, f"client_worker_{os.getpid()}", 
                {responses_stream: '>'}, 
                count=20, block=100
            )
            
            # 定期日志，监控长时间等待
            if time.time() - wait_start_time > 30:
                logging.info(f"[{os.getpid()}](PLA) 已等待Memory响应超过30秒，请求ID: {request_id}")
                wait_start_time = time.time()  # 重置计时器，避免日志刷屏

            if not responses:
                # block超时，没有读到任何消息，继续下一次循环等待
                continue
            
            for stream, message_list in responses:
                for message_id, data in message_list:
                    try:
                        resp_data = json.loads(data.get('data', '{}'))
                        resp_request_id = resp_data.get('request_id')

                        # 检查是否是我们期望的响应
                        if resp_request_id == request_id:
                            # 是我们等待的响应
                            memory_response = resp_data
                            
                            # 确认消息已处理
                            redis_conn.xack(responses_stream, group_name, message_id)
                            
                            logging.info(f"[{os.getpid()}](PLA) 收到匹配的Memory响应，请求ID: {request_id}，总等待时间: {time.time() - wait_start_time:.2f}秒")
                            
                            # 已找到响应，跳出循环
                            break
                        else:
                            # 不是我们等待的响应，忽略它
                            pass

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}](PLA) 无法解析或处理Memory响应消息 (ID: {message_id}): {e}。确认此消息以防死循环。")
                        # 对于无法解析的消息，应该确认，防止反复处理
                        redis_conn.xack(responses_stream, group_name, message_id)
                        continue
                
                if memory_response:
                    break  # 跳出外层for循环
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}](PLA) 等待Memory响应时发生错误: {e}，1秒后重试...")
            time.sleep(1)
    
    # 检查响应状态
    if not memory_response or memory_response.get('status') != 'success':
        logging.warning(f"[{os.getpid()}](PLA) 未收到有效Memory响应或请求失败")
        return []
    
    # 提取并返回搜索结果
    return memory_response.get('data', [])


def update(redis_conn, text, image=None):
    """
    向Memory Service发送更新请求并等待响应
    
    Args:
        redis_conn: Redis连接对象
        text: 要添加的文本描述
        image: PIL图像对象或None
    
    Returns:
        bool: 操作是否成功
    """
    # 编码图像
    image_data = encode_image(image) if image else None
    
    # 创建请求
    request_id = str(uuid.uuid4())
    request = {
        "id": request_id,
        "operation": "update",
        "text": text,
        "image_data": image_data
    }
    
    # 定义流
    requests_stream = STREAMS["memory_requests"]
    responses_stream = STREAMS["memory_responses"]
    
    # 创建消费者组(如果不存在)
    group_name = f"memory_client_{os.getpid()}"
    try:
        redis_conn.xgroup_create(responses_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PLA) Memory response group already exists: {e}")
        pass
    
    # 发送请求
    redis_conn.xadd(requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](PLA) 向Memory发送更新请求: {request_id}")
    
    # 等待响应
    memory_response = None
    wait_start_time = time.time()
    max_wait_time = 300  # 最长等待时间，单位秒

    while memory_response is None and (time.time() - wait_start_time < max_wait_time):
        try:
            # 使用block参数高效等待
            responses = redis_conn.xreadgroup(
                group_name, f"client_worker_{os.getpid()}", 
                {responses_stream: '>'}, 
                count=20, block=100
            )
            
            # 定期日志，监控长时间等待
            if time.time() - wait_start_time > 30:
                logging.info(f"[{os.getpid()}](PLA) 已等待Memory响应超过30秒，请求ID: {request_id}")
                wait_start_time = time.time()  # 重置计时器，避免日志刷屏

            if not responses:
                # block超时，没有读到任何消息，继续下一次循环等待
                continue
            
            for stream, message_list in responses:
                for message_id, data in message_list:
                    try:
                        resp_data = json.loads(data.get('data', '{}'))
                        resp_request_id = resp_data.get('request_id')

                        # 检查是否是我们期望的响应
                        if resp_request_id == request_id:
                            # 是我们等待的响应
                            memory_response = resp_data
                            
                            # 确认消息已处理
                            redis_conn.xack(responses_stream, group_name, message_id)
                            
                            logging.info(f"[{os.getpid()}](PLA) 收到匹配的Memory响应，请求ID: {request_id}，总等待时间: {time.time() - wait_start_time:.2f}秒")
                            
                            # 已找到响应，跳出循环
                            break
                        else:
                            # 不是我们等待的响应，忽略它
                            pass

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}](PLA) 无法解析或处理Memory响应消息 (ID: {message_id}): {e}。确认此消息以防死循环。")
                        # 对于无法解析的消息，应该确认，防止反复处理
                        redis_conn.xack(responses_stream, group_name, message_id)
                        continue
                
                if memory_response:
                    break  # 跳出外层for循环
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}](PLA) 等待Memory响应时发生错误: {e}，1秒后重试...")
            time.sleep(1)
    
    # 检查响应状态
    if not memory_response or memory_response.get('status') != 'success':
        logging.warning(f"[{os.getpid()}](PLA) 未收到有效Memory响应或请求失败")
        return False
    
    # 操作成功
    return True


def can_stop(redis_conn, question, rgb_im=None, must_stop=False, used_steps=0):
    """
    向Stopping Service发送请求，询问是否可以停止探索
    
    Args:
        redis_conn: Redis连接对象
        question: 问题对象
        images: 图像数据列表，可选
        must_stop: 是否达到最大步数限制，必须停止探索
        used_steps: 已使用的步数，默认为0
    
    Returns:
        dict: 停止服务的响应，包含status和confidence等信息
    """
    # 编码图像
    image_data = encode_image(rgb_im) if rgb_im else None
    
    # 创建请求
    request_id = str(uuid.uuid4())
    request = {
        "question": question,
        "image": image_data,
        "must_stop": must_stop,
        "used_steps": used_steps
    }
    
    # 定义流
    planner_to_stopping_stream = STREAMS["planner_to_stopping"]
    stopping_to_planner_stream = STREAMS["stopping_to_planner"]
    
    # 创建消费者组(如果不存在)
    group_name = f"planner_client_{os.getpid()}"
    try:
        redis_conn.xgroup_create(stopping_to_planner_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PLA) Stopping response group already exists: {e}")
        pass
    
    # 发送请求
    redis_conn.xadd(planner_to_stopping_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](PLA) 向Stopping Service发送请求: {request_id}")
    
    # 等待响应
    stopping_response = None
    wait_start_time = time.time()
    max_wait_time = 300  # 最长等待时间，单位秒

    while stopping_response is None and (time.time() - wait_start_time < max_wait_time):
        try:
            # 使用block参数高效等待
            responses = redis_conn.xreadgroup(
                group_name, f"client_worker_{os.getpid()}", 
                {stopping_to_planner_stream: '>'}, 
                count=20, block=100
            )
            
            # 定期日志，监控长时间等待
            if time.time() - wait_start_time > 30:
                logging.info(f"[{os.getpid()}](PLA) 已等待Stopping Service响应超过30秒，请求ID: {request_id}")
                wait_start_time = time.time()  # 重置计时器，避免日志刷屏

            if not responses:
                # block超时，没有读到任何消息，继续下一次循环等待
                continue
            
            for stream, message_list in responses:
                for message_id, data in message_list:
                    try:
                        resp_data = json.loads(data.get('data', '{}'))
                        
                        # 是我们等待的响应
                        stopping_response = resp_data
                        
                        # 确认消息已处理
                        redis_conn.xack(stopping_to_planner_stream, group_name, message_id)
                        
                        logging.info(f"[{os.getpid()}](PLA) 收到Stopping Service响应，总等待时间: {time.time() - wait_start_time:.2f}秒")
                        
                        # 已找到响应，跳出循环
                        break

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}](PLA) 无法解析或处理Stopping响应消息 (ID: {message_id}): {e}。确认此消息以防死循环。")
                        # 对于无法解析的消息，应该确认，防止反复处理
                        redis_conn.xack(stopping_to_planner_stream, group_name, message_id)
                        continue
                
                if stopping_response:
                    break  # 跳出外层for循环
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}](PLA) 等待Stopping响应时发生错误: {e}，1秒后重试...")
            time.sleep(1)
    
    # 检查响应状态
    if not stopping_response:
        logging.warning(f"[{os.getpid()}](PLA) 未收到有效Stopping响应")
        # 返回一个默认响应，表示继续探索
        return {"status": "continue", "confidence": 0.0}
    
    # 返回stopping service的响应
    return stopping_response


def get_group_info(redis_conn, group_id):
    """
    从Redis中读取指定组的信息
    
    Args:
        redis_conn: Redis连接对象
        group_id: 组ID
        
    Returns:
        dict: 组信息字典
    """
    group_info = {}
    
    # 读取基本信息（使用get命令存储的字符串值）
    group_info["group_id"] = redis_conn.get(f"{GROUP_INFO['group_id']}{group_id}")
    group_info["scene"] = redis_conn.get(f"{GROUP_INFO['scene']}{group_id}")
    
    angle = redis_conn.get(f"{GROUP_INFO['angle']}{group_id}")
    if angle:
        group_info["angle"] = float(angle)
    
    # 可选值，如果存在则读取
    floor = redis_conn.get(f"{GROUP_INFO['floor']}{group_id}")
    if floor:
        group_info["floor"] = floor
    
    max_steps = redis_conn.get(f"{GROUP_INFO['max_steps']}{group_id}")
    if max_steps:
        group_info["max_steps"] = int(max_steps)
    
    floor_height = redis_conn.get(f"{GROUP_INFO['floor_height']}{group_id}")
    if floor_height:
        group_info["floor_height"] = float(floor_height)
    
    scene_size = redis_conn.get(f"{GROUP_INFO['scene_size']}{group_id}")
    if scene_size:
        group_info["scene_size"] = float(scene_size)
    
    # 读取坐标信息（使用hget命令存储的哈希值）
    pts = redis_conn.hgetall(f"{GROUP_INFO['pts']}{group_id}")
    if pts:
        group_info["pts"] = {
            "x": float(pts.get("x", 0)),
            "y": float(pts.get("y", 0)),
            "z": float(pts.get("z", 0))
        }
    
    rotation_data = redis_conn.hgetall(f"{GROUP_INFO['rotation']}{group_id}")
    if rotation_data:
        group_info["rotation"] = {}
        for k, v in rotation_data.items():
            group_info["rotation"][k] = float(v)
    
    # 读取问题数量
    num_questions_init = redis_conn.get(f"{GROUP_INFO['num_questions_init']}{group_id}")
    if num_questions_init:
        group_info["num_questions_init"] = int(num_questions_init)
    
    num_questions_follow_up = redis_conn.get(f"{GROUP_INFO['num_questions_follow_up']}{group_id}")
    if num_questions_follow_up:
        group_info["num_questions_follow_up"] = int(num_questions_follow_up)
    
    # 读取答案映射
    correct_answers = redis_conn.hgetall(f"{GROUP_INFO['correct_answers']}{group_id}")
    if correct_answers:
        group_info["correct_answers"] = correct_answers
    
    return group_info


def set_group_info(redis_conn, group_id, group_info):
    """
    将组信息写入Redis
    
    Args:
        redis_conn: Redis连接对象
        group_id: 组ID
        group_info: 组信息字典
        
    Returns:
        bool: 操作是否成功
    """
    try:
        pipe = redis_conn.pipeline()
        
        # 设置基本信息
        if "group_id" in group_info:
            pipe.set(f"{GROUP_INFO['group_id']}{group_id}", group_info["group_id"])
        
        if "scene" in group_info:
            pipe.set(f"{GROUP_INFO['scene']}{group_id}", group_info["scene"])
        
        if "angle" in group_info:
            pipe.set(f"{GROUP_INFO['angle']}{group_id}", group_info["angle"])
        
        if "floor" in group_info:
            pipe.set(f"{GROUP_INFO['floor']}{group_id}", group_info["floor"])
        
        if "max_steps" in group_info:
            pipe.set(f"{GROUP_INFO['max_steps']}{group_id}", group_info["max_steps"])
        
        if "floor_height" in group_info:
            pipe.set(f"{GROUP_INFO['floor_height']}{group_id}", group_info["floor_height"])
        
        if "scene_size" in group_info:
            pipe.set(f"{GROUP_INFO['scene_size']}{group_id}", group_info["scene_size"])
        
        # 设置坐标信息
        if "pts" in group_info and isinstance(group_info["pts"], dict):
            pipe.hset(f"{GROUP_INFO['pts']}{group_id}", mapping=group_info["pts"])
        
        if "rotation" in group_info and isinstance(group_info["rotation"], dict):
            pipe.hset(f"{GROUP_INFO['rotation']}{group_id}", mapping=group_info["rotation"])
        elif "rotation" in group_info and isinstance(group_info["rotation"], list):
            # 如果rotation是列表，转换为字典格式
            rotation_dict = {str(i): val for i, val in enumerate(group_info["rotation"])}
            pipe.hset(f"{GROUP_INFO['rotation']}{group_id}", mapping=rotation_dict)
        
        # 设置问题数量
        if "num_questions_init" in group_info:
            pipe.set(f"{GROUP_INFO['num_questions_init']}{group_id}", group_info["num_questions_init"])
        
        if "num_questions_follow_up" in group_info:
            pipe.set(f"{GROUP_INFO['num_questions_follow_up']}{group_id}", group_info["num_questions_follow_up"])
        
        # 设置答案映射
        if "correct_answers" in group_info and isinstance(group_info["correct_answers"], dict):
            pipe.hset(f"{GROUP_INFO['correct_answers']}{group_id}", mapping=group_info["correct_answers"])
        
        # 执行所有命令
        pipe.execute()
        return True
    
    except Exception as e:
        logging.error(f"[{os.getpid()}](PLA) 设置组信息时出错: {e}")
        return False


# ===================================
# ---------- ParaEQA Class ----------
# ===================================

class ParaEQA:
    def __init__(self, config):
        self.config = config
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.camera_tilt = self.config.get("camera_tilt_deg", -30) * np.pi / 180
        self.cam_intr = get_cam_intr(
            self.config.get("hfov", 120), 
            self.config.get("img_height", 480), 
            self.config.get("img_width", 640)
        )
        self.img_height = self.config.get("img_height", 480)
        self.img_width = self.config.get("img_width", 640)

        self.simulator = None
        
        self.use_parallel = self.config.get("use_parallel", True)
        self.use_rag = self.config.get("memory", {}).get("use_rag", True)

        # init prompts
        prompt = self.config.get("prompt", {}).get("planner", {})
        self.prompt_caption = prompt.get("caption", "")
        self.prompt_rel = prompt.get("relevent", "")
        self.prompt_question = prompt.get("question", "")
        self.prompt_lsv = prompt.get("local_sem", "")
        self.prompt_gsv = prompt.get("global_sem", "")
        
        # init VLM model
        config_vlm = config.get("vlm", {}).get("planner", {})
        model_name = config_vlm.get("model", "qwen/qwen2.5-vl-72b-instruct")
        server = config_vlm.get("server", "openrouter")
        base_url = config_vlm.get("base_url", None)
        api_key = config_vlm.get("api_key", None)
        self.vlm = VLM_API(model_name, server, base_url, api_key)
        
        config_vlm_lite = config.get("vlm", {}).get("planner_lite", {})
        model_name_lite = config_vlm_lite.get("model", "qwen/qwen2.5-vl-72b-instruct")
        server_lite = config_vlm_lite.get("server", "openrouter")
        base_url_lite = config_vlm_lite.get("base_url", None)
        api_key_lite = config_vlm_lite.get("api_key", None)
        self.vlm_lite = VLM_API(model_name_lite, server_lite, base_url_lite, api_key_lite)
        
        config_vlm_tiny = config.get("vlm", {}).get("planner_tiny", {})
        model_name_tiny = config_vlm_tiny.get("model", "qwen/qwen2.5-vl-32b-instruct")
        server_tiny = config_vlm_tiny.get("server", "openrouter")
        base_url_tiny = config_vlm_tiny.get("base_url", None)
        api_key_tiny = config_vlm_tiny.get("api_key", None)
        self.vlm_tiny = VLM_API(model_name_tiny, server_tiny, base_url_tiny, api_key_tiny)
        
        # 连接Redis
        self.redis_conn = get_redis_connection(config)
        
        # init detector 'yolov12{n/s/m/l/x}.pt'
        self.detector = YOLO(self.config.get("detector", "./data/yolo11x.pt"))

        # init drawing
        self.letters = ["A", "B", "C", "D"]  # always four
        self.fnt = ImageFont.truetype("data/Open_Sans/static/OpenSans-Regular.ttf", 30,)

        self.confident_threshold = ["c", "d", "e", "yes"]

    
    def init_sim(self, scene):
        # Set up scene in Habitat
        try:
            self.simulator.close()
        except:
            pass
        
        scene_data_path = self.config.get("scene_data_path", "./data/HM3D")
        scene_mesh_dir = os.path.join(
            scene_data_path, scene, scene[6:] + ".basis" + ".glb"
        )
        navmesh_file = os.path.join(
            scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
        )
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": self.config.get("camera_height", 1.5),
            "width": self.img_width,
            "height": self.img_height,
            "hfov": self.config.get("hfov", 120),
        }
        sim_config = make_simple_config(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_config)
        pathfinder = self.simulator.pathfinder
        pathfinder.seed(self.config.get("seed", 42))
        pathfinder.load_nav_mesh(navmesh_file)
        
        if not pathfinder.is_loaded:
            logging.error("Not loaded .navmesh file yet. Please check file path {}.".format(navmesh_file))

        agent = self.simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()

        return agent, agent_state, self.simulator, pathfinder

    
    def init_planner(self, tsdf_bnds, pts):
        # Initialize TSDF Planner
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=self.config.get("tsdf_grid_size", 0.1),
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=self.config.get("init_clearance", 0.5) * 2,
        )
        return tsdf_planner
    
    
    def prepare_data(self, question_data, question_ind):
        # 从 Redis 获取当前 group_id
        group_id = get_current_group_id(self.redis_conn)
        if not group_id:
            raise ValueError("无法找到当前活跃的 group_id")
        
        # 从 GROUP_INFO 获取组信息
        group_info = get_group_info(self.redis_conn, group_id)
        
        # 从 description 提取问题和选项
        description = question_data.get('description', '')
        
        # 使用正则表达式提取问题和选项
        match = re.match(r'(.*?)\s*A\)(.*?)\s*B\)(.*?)\s*C\)(.*?)\s*D\)(.*?)(?:\.|\s*$)', description, re.DOTALL)
        if match:
            question = match.group(1).strip()
            choices = [
                match.group(2).strip(),
                match.group(3).strip(),
                match.group(4).strip(),
                match.group(5).strip()
            ]
        else:
            # 如果无法匹配，报告错误
            raise ValueError(f"无法从描述中提取问题和选项: {description}")
        
        # 将选项格式化为字符串列表
        # choices = str(choices)
        
        # 获取答案 (A、B、C、D)
        answer = question_data.get('answer', None)
        if answer is None:
            raise ValueError("问题数据中未提供答案")
        
        # 从 GROUP_INFO 获取场景和楼层信息
        scene = group_info.get('scene')
        floor = group_info.get('floor', '0')  # 默认为 0
        scene_floor = scene + "_" + floor
        
        # 获取初始位置和角度
        if 'pts' in group_info:
            pts = [
                float(group_info['pts'].get('x', 0)),
                float(group_info['pts'].get('y', 0)),
                float(group_info['pts'].get('z', 0))
            ]
        else:
            # 如果没有在 GROUP_INFO 中找到，使用默认值
            pts = self.init_pose_data[scene_floor]["init_pts"]
        
        angle = float(group_info.get('angle', 0))
        
        logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")

        # Re-format the question to follow LLaMA style
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]

        # open or close vocab
        is_open_vocab = False
        if is_open_vocab:
            answer_index = vlm_pred_candidates.index(answer)
            if 0 <= answer_index < len(choices):
                answer = choices[answer_index]
        else:
            for token, choice in zip(vlm_pred_candidates, choices):
                vlm_question += "\n" + token + ". " + choice
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(self.config.get("output_dir", "logs"), str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)

        agent, agent_state, self.simulator, pathfinder = self.init_sim(scene)
        
        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
        ).tolist()
        pts_normal = pos_habitat_to_normal(np.array(pts))
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * self.config.get("max_step_room_size_ratio", 3))
        
        question_data["max_steps"] = num_step
        
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )

        # init planner
        tsdf_planner = self.init_planner(tsdf_bnds, pts_normal)

        metadata = {
            "question_ind": question_ind,
            "org_question": question,
            "question": vlm_question,
            "answer": answer,
            "scene": scene,
            "floor": floor,
            "max_steps": num_step,
            "angle": angle,
            "init_pts": pts,
            "init_rotation": rotation,
            "floor_height": floor_height,
            "scene_size": scene_size,
        }

        return metadata, agent, agent_state, tsdf_planner, episode_data_dir


    # 输入的question_data类似下面这样：
    """
    {'scene': '00797-99ML7CGPqsQ', 'floor': '0', 'question': 'Is the door color darker than the ceiling color?', 'choices': "['Yes', 'No', 'They are the same color', 'The ceiling is darker']", 'question_formatted': 'Is the door color darker than the ceiling color? A) Yes B) No C) They are the same color D) The ceiling is darker. Answer:', 'answer': 'A', 'label': 'Comparison', 'source_image': '00797-99ML7CGPqsQ_0.png'}
    """
    def run(self, question_data, question_ind):
        # 从 Redis 获取当前 group_id
        group_id = get_current_group_id(self.redis_conn)
        if not group_id:
            raise ValueError("无法找到当前活跃的 group_id")
        
        # 从 GROUP_INFO 获取组信息
        group_info = get_group_info(self.redis_conn, group_id)
        
        # 在开始探索之前，先询问stopping service是否可以直接回答问题
        stopping_response = can_stop(self.redis_conn, question_data)
        if stopping_response.get("status") == "stop":
            # 可以直接回答问题，无需探索
            logging.info(f"[{os.getpid()}](PLA) Stopping Service决定直接回答问题，置信度: {stopping_response.get('confidence', 0.0)}")
            # 创建一个基本的结果对象
            result = {
                "meta": {
                    "question_ind": question_ind,
                    "org_question": question_data.get("description", ""),
                    "answer": question_data.get("answer", ""),
                    "scene": group_info.get("scene", ""),
                    "floor": group_info.get("floor", ""),
                },
                "step": [],
                "summary": {
                    "explored_steps": 0,
                },
            }
            
            # 直接回答问题后，不需要更新GROUP_INFO
            return result
        
        # 准备数据，开始新的探索
        meta, agent, agent_state, tsdf_planner, episode_data_dir = self.prepare_data(question_data, question_ind)

        result = {
            "meta": meta,
            "step": [],
            "summary": {},
        }

        # Extract metadata
        question = meta["org_question"]
        vlm_question = meta["question"]
        answer = meta["answer"]
        scene = meta["scene"]
        floor = meta["floor"]
        num_step = meta["max_steps"]
        angle = meta["angle"]
        pts = np.array(meta["init_pts"])
        rotation = meta["init_rotation"]
        floor_height = meta["floor_height"]
        scene_size = meta["scene_size"]

        # Run steps
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        
        for cnt_step in range(num_step):
            logging.info(f"\n== step: {cnt_step}")
            
            # Initialize kb
            kb = []

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")

            agent_state.position = pts
            agent_state.rotation = rotation
            agent.set_state(agent_state)

            pts_normal = pos_habitat_to_normal(pts)

            result["step"].append({"step": cnt_step, "pts": pts.tolist(), "angle": angle})

            # Update camera info
            sensor = agent.get_state().sensor_states["depth_sensor"]
            quaternion_0 = sensor.rotation
            translation_0 = sensor.position

            cam_pose = np.eye(4)
            cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
            cam_pose[:3, 3] = translation_0
            cam_pose_normal = pose_habitat_to_normal(cam_pose)
            cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            obs = self.simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]

            rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
            
            if self.use_rag:
                # "What room are you most likely to be in at the moment? Answer with a phrase"
                room = self.vlm_lite.request_with_retry(image=rgb_im, prompt="What room are you most likely to be in at the moment? Answer with a phrase")

                objects = self.detector(rgb_im)[0]
                objs_info = []
                for box in objects.boxes:
                    cls = objects.names[box.cls.item()]
                    box = box.xyxy[0].cpu()
                    
                    # 裁剪目标区域进行描述
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    obj_im = rgb_im.crop((x1, y1, x2, y2))
                    # "Describe this image."
                    obj_caption = self.vlm_tiny.request_with_retry(image=obj_im, prompt=self.prompt_caption)
                    
                    # 中心点转换世界坐标
                    x, y = (x1 + x2) / 2, (y1 + y2) / 2
                    world_pos = pixel2world(x, y, depth[int(y), int(x)], cam_pose)
                    world_pos = pos_normal_to_habitat(world_pos)
                    
                    # 保存目标信息
                    objs_info.append({"room": room, "cls": cls ,"caption": obj_caption[0], "pos": world_pos.tolist()})

                # "Describe this image."
                caption = self.vlm_lite.request_with_retry(image=rgb_im, prompt=self.prompt_caption)

                if self.config.get("save_obs", True):
                    save_rgbd(rgb, depth, os.path.join(episode_data_dir, f"{cnt_step}_rgbd.png"))
                    rgb_path = os.path.join(episode_data_dir, "{}.png".format(cnt_step))
                    plt.imsave(rgb_path, rgb)
                
                # 构建目标信息
                objs_str = json.dumps(objs_info)
                
                # 向Memory添加知识
                update(self.redis_conn, f"{step_name}: agent position is {pts}. {caption}. Objects: {objs_str}", rgb_im)

            num_black_pixels = np.sum(
                np.sum(rgb, axis=-1) == 0
            )  # sum over channel first
            
            if num_black_pixels < self.config.get("black_pixel_ratio", 0.7) * self.img_width * self.img_height:
                # TSDF fusion
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=self.cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(self.config.get("margin_h_ratio", 0.6) * self.img_height),
                    margin_w=int(self.config.get("margin_w_ratio", 0.25) * self.img_width),
                )

                # 在每一步后询问stopping service是否可以停止探索
                # 如果可以结束，把更新后的信息存储到GROUP_INFO中
                # "... How confident are you in answering this question from your current perspective? ..."
                stopping_response = can_stop(self.redis_conn, question_data, rgb_im, used_steps=cnt_step + 1)
                if stopping_response.get("status") == "stop":
                    # 可以停止探索，结束循环
                    logging.info(f"[{os.getpid()}](PLA) Stopping Service决定停止探索，置信度: {stopping_response.get('confidence', 0.0)}")
                    break

                # Get frontier candidates
                prompt_points_pix = []
                if self.config.get("use_active", True):
                    prompt_points_pix, fig = (
                        tsdf_planner.find_prompt_points_within_view(
                            pts_normal,
                            self.img_width,
                            self.img_height,
                            self.cam_intr,
                            cam_pose_tsdf,
                            **self.config.get("visual_prompt", {})
                        )
                    )
                    fig.tight_layout()
                    plt.savefig(os.path.join(episode_data_dir, "prompt_points.png".format(cnt_step)))
                    plt.close()

                # Visual prompting
                actual_num_prompt_points = len(prompt_points_pix)
                if actual_num_prompt_points >= self.config.get("visual_prompt", {}).get("min_num_prompt_points", 2):
                    rgb_im_draw = draw_letters(rgb_im, 
                                            prompt_points_pix, 
                                            self.letters, 
                                            self.config.get("visual_prompt", {}).get("circle_radius", 18), 
                                            self.fnt, 
                                            os.path.join(episode_data_dir, f"{cnt_step}_draw.png"))

                    # get VLM reasoning for exploring
                    if self.config.get("use_lsv", True):
                        response = None
                        
                        if self.use_rag:
                            kb = search(
                                self.redis_conn, 
                                self.prompt_lsv.format(question), 
                                rgb_im, 
                                top_k=self.config.get("memory", {}).get("max_retrieval_num", 5) if cnt_step > self.config.get("memory", {}).get("max_retrieval_num", 5) else cnt_step
                            )
                            
                            # "... Which direction (black letters on the image) would you explore then? ..."
                            response = self.vlm.request_with_retry(rgb_im_draw, self.prompt_lsv.format(question), kb)[0]
                        
                        else:
                            prompt_lsv_no_rag = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Answer with a single letter."
                            
                            response = self.vlm.request_with_retry(rgb_im_draw, prompt_lsv_no_rag)[0]
                        
                        lsv = np.zeros(actual_num_prompt_points)
                        for i in range(actual_num_prompt_points):
                            if response == self.letters[i]:
                                lsv[i] = 1
                        lsv *= actual_num_prompt_points / 3
                    
                    else:
                        lsv = (
                            np.ones(actual_num_prompt_points) / actual_num_prompt_points
                        )

                    # base - use image without label
                    if self.config.get("use_gsv", True):
                        response = None
                        
                        if self.use_rag:
                            kb = search(
                                self.redis_conn, 
                                self.prompt_gsv.format(question), 
                                rgb_im, 
                                top_k=self.config.get("memory", {}).get("max_retrieval_num", 5) if cnt_step > self.config.get("memory", {}).get("max_retrieval_num", 5) else cnt_step
                            )
                            
                            # "... Is there any direction shown in the image worth exploring? ..."
                            response = self.vlm.request_with_retry(rgb_im, self.prompt_gsv.format(question), kb)[0].strip(".")
                        
                        else:
                            prompt_gsv_no_rag = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                            
                            response = self.vlm.request_with_retry(rgb_im, prompt_gsv_no_rag)[0].strip(".")
                        
                        gsv = np.zeros(2)
                        if response == "Yes":
                            gsv[0] = 1
                        else:
                            gsv[1] = 1
                        gsv = (np.exp(gsv[0] / self.config.get("gsv_T", 0.5)) / self.config.get("gsv_F", 3))  # scale before combined with lsv
                    
                    else:
                        gsv = 1
                    
                    sv = lsv * gsv
                    logging.info(f"Exp - LSV: {lsv} GSV: {gsv} SV: {sv}")

                    # Integrate semantics only if there is any prompted point
                    tsdf_planner.integrate_sem(
                        sem_pix=sv,
                        radius=1.0,
                        obs_weight=1.0,
                    )  # voxel locations already saved in tsdf class

            else:
                logging.info("Skipping black image!")

            # Determine next point
            if cnt_step < num_step - 1:  # 避免在最后一步计算下一个位置
                pts_normal, angle, cur_angle, pts_pix, fig = tsdf_planner.find_next_pose(
                    pts=pts_normal,
                    angle=angle,
                    cam_pose=cam_pose_tsdf,
                    flag_no_val_weight=cnt_step < self.config.get("min_random_init_steps", 2),
                    **self.config.get("planner", {})
                )
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)

                # Add path to ax5, with colormap to indicate order
                ax5 = fig.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                fig.tight_layout()

                plt.savefig(
                    os.path.join(episode_data_dir, "map.png".format(cnt_step + 1))
                )
                plt.close()
                
            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
            ).tolist()
            
            # 当达到最大探索步数时，强制结束探索
            if cnt_step == num_step - 1:
                logging.info(f"达到最大探索步数 {num_step}，强制结束探索")
                stopping_response = can_stop(self.redis_conn, question_data, rgb_im, must_stop=True, used_steps=cnt_step + 1)
                if stopping_response.get("status") == "stop":
                    # 可以停止探索，结束循环
                    logging.info(f"[{os.getpid()}](PLA) Stopping Service决定停止探索，置信度设置为1.0")
                else:
                    raise RuntimeError(
                        f"达到最大探索步数 {num_step}，但Stopping Service未能确认停止探索"
                    )
                
                

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Scene: {scene}, Floor: {floor}")
        logging.info(f"Explored steps: {cnt_step + 1}")
        
        # 记录探索的步数
        result["summary"]["explored_steps"] = cnt_step + 1
        
        # 更新 GROUP_INFO
        updated_group_info = {
            'group_id': group_id,
            'scene': scene,
            'floor': floor,
            'angle': angle,
            'pts': {
                'x': float(pts[0]),
                'y': float(pts[1]),
                'z': float(pts[2])
            },
            'rotation': {str(i): float(val) for i, val in enumerate(rotation)},
            'floor_height': float(floor_height),
            'scene_size': float(scene_size),
            'num_questions_init': int(group_info.get('num_questions_init', 0)),
            'num_questions_follow_up': int(group_info.get('num_questions_follow_up', 0)),
            'correct_answers': group_info.get('correct_answers', {})
        }
        
        # 只有当 use_parallel 为 True 时才更新 GROUP_INFO
        if self.use_parallel:
            set_group_info(self.redis_conn, group_id, updated_group_info)

        return result
