# utils/para_eaq.py

# TODO: group_info相关问题
# FIXME: vlm相关问题

import os
import json
import numpy as np
import csv
import pickle
import logging
import math
import quaternion
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import uuid
import time
import base64
from io import BytesIO

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
# from utils.vlm_local import VLM_Local
from common.redis_client import get_redis_connection, STREAMS
from utils.image_processor import encode_image

np.set_printoptions(precision=3)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"


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
        logging.info(f"[{os.getpid()}] Memory response group already exists: {e}")
    
    # 发送请求
    redis_conn.xadd(requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}] 向Memory发送搜索请求: {request_id}")
    
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
                logging.info(f"[{os.getpid()}] 已等待Memory响应超过30秒，请求ID: {request_id}")
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
                            
                            logging.info(f"[{os.getpid()}] 收到匹配的Memory响应，请求ID: {request_id}，总等待时间: {time.time() - wait_start_time:.2f}秒")
                            
                            # 已找到响应，跳出循环
                            break
                        else:
                            # 不是我们等待的响应，忽略它
                            pass

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}] 无法解析或处理Memory响应消息 (ID: {message_id}): {e}。确认此消息以防死循环。")
                        # 对于无法解析的消息，应该确认，防止反复处理
                        redis_conn.xack(responses_stream, group_name, message_id)
                        continue
                
                if memory_response:
                    break  # 跳出外层for循环
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}] 等待Memory响应时发生错误: {e}，1秒后重试...")
            time.sleep(1)
    
    # 检查响应状态
    if not memory_response or memory_response.get('status') != 'success':
        logging.warning(f"[{os.getpid()}] 未收到有效Memory响应或请求失败")
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
        logging.info(f"[{os.getpid()}] Memory response group already exists: {e}")
    
    # 发送请求
    redis_conn.xadd(requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}] 向Memory发送更新请求: {request_id}")
    
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
                logging.info(f"[{os.getpid()}] 已等待Memory响应超过30秒，请求ID: {request_id}")
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
                            
                            logging.info(f"[{os.getpid()}] 收到匹配的Memory响应，请求ID: {request_id}，总等待时间: {time.time() - wait_start_time:.2f}秒")
                            
                            # 已找到响应，跳出循环
                            break
                        else:
                            # 不是我们等待的响应，忽略它
                            pass

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}] 无法解析或处理Memory响应消息 (ID: {message_id}): {e}。确认此消息以防死循环。")
                        # 对于无法解析的消息，应该确认，防止反复处理
                        redis_conn.xack(responses_stream, group_name, message_id)
                        continue
                
                if memory_response:
                    break  # 跳出外层for循环
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}] 等待Memory响应时发生错误: {e}，1秒后重试...")
            time.sleep(1)
    
    # 检查响应状态
    if not memory_response or memory_response.get('status') != 'success':
        logging.warning(f"[{os.getpid()}] 未收到有效Memory响应或请求失败")
        return False
    
    # 操作成功
    return True


def can_stop(redis_conn, question, images=None):
    """
    向Stopping Service发送请求，询问是否可以停止探索
    
    Args:
        redis_conn: Redis连接对象
        question: 问题对象
        images: 图像数据列表，可选
    
    Returns:
        dict: 停止服务的响应，包含status和confidence等信息
    """
    # 创建请求
    request_id = str(uuid.uuid4())
    request = {
        "question": question,
        "images": images or []
    }
    
    # 定义流
    planner_to_stopping_stream = STREAMS["planner_to_stopping"]
    stopping_to_planner_stream = STREAMS["stopping_to_planner"]
    
    # 创建消费者组(如果不存在)
    group_name = f"planner_client_{os.getpid()}"
    try:
        redis_conn.xgroup_create(stopping_to_planner_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        logging.info(f"[{os.getpid()}] Stopping response group already exists: {e}")
    
    # 发送请求
    redis_conn.xadd(planner_to_stopping_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}] 向Stopping Service发送请求: {request_id}")
    
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
                logging.info(f"[{os.getpid()}] 已等待Stopping Service响应超过30秒，请求ID: {request_id}")
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
                        
                        logging.info(f"[{os.getpid()}] 收到Stopping Service响应，总等待时间: {time.time() - wait_start_time:.2f}秒")
                        
                        # 已找到响应，跳出循环
                        break

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}] 无法解析或处理Stopping响应消息 (ID: {message_id}): {e}。确认此消息以防死循环。")
                        # 对于无法解析的消息，应该确认，防止反复处理
                        redis_conn.xack(stopping_to_planner_stream, group_name, message_id)
                        continue
                
                if stopping_response:
                    break  # 跳出外层for循环
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}] 等待Stopping响应时发生错误: {e}，1秒后重试...")
            time.sleep(1)
    
    # 检查响应状态
    if not stopping_response:
        logging.warning(f"[{os.getpid()}] 未收到有效Stopping响应")
        # 返回一个默认响应，表示继续探索
        return {"status": "continue", "confidence": 0.0}
    
    # 返回stopping service的响应
    return stopping_response


class ParaEQA:
    def __init__(self, config, group_info, gpu_id):
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

        # init prompts
        prompt = self.config.get("prompt", {}).get("planner", {})
        self.prompt_caption = prompt.get("caption", "")
        self.prompt_rel = prompt.get("relevent", "")
        self.prompt_question = prompt.get("question", "")
        self.prompt_lsv = prompt.get("local_sem", "")
        self.prompt_gsv = prompt.get("global_sem", "")

        # load init pose data
        with open(self.config.get("init_pose_data_path", "./data/scene_init_poses_all.csv")) as f:
            self.init_pose_data = {}
            for row in csv.DictReader(f, skipinitialspace=True):
                self.init_pose_data[row["scene_floor"]] = {
                    "init_pts": [
                        float(row["init_x"]),
                        float(row["init_y"]),
                        float(row["init_z"]),
                    ],
                    "init_angle": float(row["init_angle"]),
                }

        # init VLM model
        model_name = self.config.get("vlm", {}).get("model_api", "gpt-4.1")
        use_openrouter = self.config.get("vlm", {}).get("use_openrouter", False)
        self.vlm = VLM_API(model_name, use_openrouter)
        
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
        
        # TODO: 从Redis的group_info中获取
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

    
    # TODO: 修改prepare_data函数，适应新的question_data格式，同时从Redis中读取group_info。保证修改后prepare_data函数的返回值和当前相同。
    def prepare_data(self, question_data, question_ind):
        # Extract question
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        choices = [c.strip("'\"") for c in question_data["choices"].strip("[]").split(", ")]
        answer = question_data["answer"]

        # TODO: init_pts需要为上一个问题结束探索的位置
        init_pts = self.init_pose_data[scene_floor]["init_pts"]
        init_angle = self.init_pose_data[scene_floor]["init_angle"]
        
        logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")

        # Re-format the question to follow LLaMA style
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]

        # open or close vocab
        is_open_vocab = False
        if is_open_vocab:
            answer = choices[vlm_pred_candidates.index(answer)]
        else:
            for token, choice in zip(vlm_pred_candidates, choices):
                vlm_question += "\n" + token + ". " + choice
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(self.config.get("output_dir", "results"), str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)

        agent, agent_state, self.simulator, pathfinder = self.init_sim(scene)
        
        pts = np.array(init_pts)
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
        ).tolist()
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * self.config.get("max_step_room_size_ratio", 3))
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
            "init_pts": pts.tolist(),
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
        # 在开始探索之前，先询问stopping service是否可以直接回答问题
        stopping_response = can_stop(self.redis_conn, question_data)
        if stopping_response.get("status") == "stop":
            # 可以直接回答问题，无需探索
            logging.info(f"[{os.getpid()}] Stopping Service决定直接回答问题，置信度: {stopping_response.get('confidence', 0.0)}")
            # 创建一个基本的结果对象
            result = {
                "meta": {
                    "question_ind": question_ind,
                    "org_question": question_data.get("question", ""),
                    "answer": question_data.get("answer", ""),
                    "scene": question_data.get("scene", ""),
                    "floor": question_data.get("floor", ""),
                },
                "step": [],
                "summary": {
                    "explored_steps": 0,
                },
            }
            return result
        
        # TODO: 修改prepare_data函数，适应新的question_data格式，同时从Redis中读取group_info。保证修改后prepare_data函数的返回值和当前相同。
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
        collected_images = []  # 存储探索过程中的图像
        
        # TODO: agent.set_state() 需要衔接前一个问题
        for cnt_step in range(num_step):
            logging.info(f"\n== step: {cnt_step}")

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
            
            # "What room are you most likely to be in at the moment? Answer with a phrase"
            room = self.vlm.request_with_retry(image=rgb_im, prompt="What room are you most likely to be in at the moment? Answer with a phrase")

            objects = self.detector(rgb_im)[0]
            objs_info = []
            for box in objects.boxes:
                cls = objects.names[box.cls.item()]
                box = box.xyxy[0].cpu()
                
                # 裁剪目标区域进行描述
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                obj_im = rgb_im.crop((x1, y1, x2, y2))
                # "Describe this image."
                obj_caption = self.vlm.request_with_retry(image=obj_im, prompt=self.prompt_caption)
                
                # 中心点转换世界坐标
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                world_pos = pixel2world(x, y, depth[int(y), int(x)], cam_pose)
                world_pos = pos_normal_to_habitat(world_pos)
                
                # 保存目标信息
                objs_info.append({"room": room, "cls": cls ,"caption": obj_caption[0], "pos": world_pos.tolist()})

            if self.config.get("memory", {}).get("use_rag", True):
                # "Describe this image."
                caption = self.vlm.request_with_retry(image=rgb_im, prompt=self.prompt_caption)

            if self.config.get("save_obs", True):
                save_rgbd(rgb, depth, os.path.join(episode_data_dir, f"{cnt_step}_rgbd.png"))
                if self.config.get("memory", {}).get("use_rag", True):
                    rgb_path = os.path.join(episode_data_dir, "{}.png".format(cnt_step))
                    plt.imsave(rgb_path, rgb)
                    # 构建目标信息
                    objs_str = json.dumps(objs_info)
                    # 向Memory添加知识
                    update(self.redis_conn, f"{step_name}: agent position is {pts}. {caption}. Objects: {objs_str}", rgb_im)
                    
                    # 添加当前图像到收集列表，供stopping service使用
                    encoded_image = encode_image(rgb_im)
                    if encoded_image:
                        collected_images.append(encoded_image)

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
                stopping_response = can_stop(self.redis_conn, question_data, collected_images)
                if stopping_response.get("status") == "stop":
                    # 可以停止探索，结束循环
                    logging.info(f"[{os.getpid()}] Stopping Service决定停止探索，置信度: {stopping_response.get('confidence', 0.0)}")
                    break

                # 对于过程性的探索判断，仍然保留原来的代码
                if self.config.get("memory", {}).get("use_rag", True):
                    kb = search(
                        self.redis_conn, 
                        self.prompt_rel.format(question), 
                        rgb_im, 
                        top_k=self.config.get("memory", {}).get("max_retrieval_num", 5) if cnt_step > self.config.get("memory", {}).get("max_retrieval_num", 5) else cnt_step
                    )
                
                # "... How confident are you in answering this question from your current perspective? ..."
                # FIXME:
                smx_vlm_rel = self.vlm.get_response(rgb_im, self.prompt_rel.format(question), kb, device=self.device)[0].strip(".")
                
                logging.info(f"Rel - Prob: {smx_vlm_rel}")

                logging.info(f"Prompt Pred: {self.prompt_question.format(vlm_question)}")
                if self.config.get("memory", {}).get("use_rag", True):
                    kb = search(
                        self.redis_conn, 
                        self.prompt_question.format(vlm_question), 
                        rgb_im, 
                        top_k=self.config.get("memory", {}).get("max_retrieval_num", 5) if cnt_step > self.config.get("memory", {}).get("max_retrieval_num", 5) else cnt_step
                    )
                
                # "... Answer with the option's letter from the given choices directly. ..."
                # FIXME:
                smx_vlm_pred = self.vlm.get_response(rgb_im, self.prompt_question.format(vlm_question), kb, device=self.device)[0].strip(".")
                
                logging.info(f"Pred - Prob: {smx_vlm_pred}")

                # save data
                result["step"][cnt_step]["smx_vlm_rel"] = smx_vlm_rel[0]
                result["step"][cnt_step]["smx_vlm_pred"] = smx_vlm_pred[0]
                result["step"][cnt_step]["is_success"] = smx_vlm_pred[0] == answer

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
                        if self.config.get("memory", {}).get("use_rag", True):
                            kb = search(
                                self.redis_conn, 
                                self.prompt_lsv.format(question), 
                                rgb_im, 
                                top_k=self.config.get("memory", {}).get("max_retrieval_num", 5) if cnt_step > self.config.get("memory", {}).get("max_retrieval_num", 5) else cnt_step
                            )
                        
                        # "... Which direction (black letters on the image) would you explore then? ..."
                        # FIXME:
                        response = self.vlm.get_response(rgb_im_draw, self.prompt_lsv.format(question), kb, device=self.device)[0]
                        
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
                        if self.config.get("memory", {}).get("use_rag", True):
                            kb = search(
                                self.redis_conn, 
                                self.prompt_gsv.format(question), 
                                rgb_im, 
                                top_k=self.config.get("memory", {}).get("max_retrieval_num", 5) if cnt_step > self.config.get("memory", {}).get("max_retrieval_num", 5) else cnt_step
                            )
                        
                        # "... Is there any direction shown in the image worth exploring? ..."
                        # FIXME:
                        response = self.vlm.get_response(rgb_im, self.prompt_gsv.format(question), kb, device=self.device)[0].strip(".")
                        
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

        # 删除原来的最终回答部分，由stopping service处理

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Scene: {scene}, Floor: {floor}")
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")
        logging.info(f"Explored steps: {cnt_step + 1}")
        
        # 记录探索的步数
        result["summary"]["explored_steps"] = cnt_step + 1

        return result
