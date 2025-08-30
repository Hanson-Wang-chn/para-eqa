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
import concurrent.futures

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
    Send search request to Memory Service and wait for response
    
    Args:
        redis_conn: Redis connection object
        text: Query text
        image: PIL image object or None
        top_k: Maximum number of results to return
    
    Returns:
        list: Knowledge base search results list
    """
    # Encode image
    image_data = encode_image(image) if image else None
    
    # Create request
    request_id = str(uuid.uuid4())
    request = {
        "id": request_id,
        "operation": "search",
        "text": text,
        "image_data": image_data,
        "top_k": top_k
    }
    
    # Define streams
    requests_stream = STREAMS["memory_requests"]
    responses_stream = STREAMS["memory_responses"]
    
    # Create consumer group (if not exists)
    group_name = f"memory_client_{os.getpid()}"
    try:
        redis_conn.xgroup_create(responses_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PLA) Memory response group already exists: {e}")
        pass
    
    # Send request
    redis_conn.xadd(requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](PLA) Sending search request to Memory: {request_id}")
    
    # Wait for response
    memory_response = None
    wait_start_time = time.time()
    max_wait_time = 300  # Maximum wait time in seconds

    while memory_response is None and (time.time() - wait_start_time < max_wait_time):
        try:
            # Use block parameter for efficient waiting
            responses = redis_conn.xreadgroup(
                group_name, f"client_worker_{os.getpid()}", 
                {responses_stream: '>'}, 
                count=20, block=100
            )
            
            # Periodic logging to monitor long waits
            if time.time() - wait_start_time > 30:
                logging.info(f"[{os.getpid()}](PLA) Waiting for Memory response over 30 seconds, request ID: {request_id}")
                wait_start_time = time.time()  # Reset timer to avoid log spam

            if not responses:
                # Block timeout, no messages read, continue to next loop iteration
                continue
            
            for stream, message_list in responses:
                for message_id, data in message_list:
                    try:
                        resp_data = json.loads(data.get('data', '{}'))
                        resp_request_id = resp_data.get('request_id')

                        # Check if this is the expected response
                        if resp_request_id == request_id:
                            # This is the response we're waiting for
                            memory_response = resp_data
                            
                            # Acknowledge message as processed
                            redis_conn.xack(responses_stream, group_name, message_id)
                            
                            logging.info(f"[{os.getpid()}](PLA) Received matching Memory response, request ID: {request_id}, total wait time: {time.time() - wait_start_time:.2f} seconds")
                            
                            # Found response, break out of loop
                            break
                        else:
                            # Not the response we're waiting for, ignore it
                            pass

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}](PLA) Unable to parse or process Memory response message (ID: {message_id}): {e}. Acknowledging this message to prevent infinite loop.")
                        # For unparseable messages, should acknowledge to prevent repeated processing
                        redis_conn.xack(responses_stream, group_name, message_id)
                        continue
                
                if memory_response:
                    break  # Break out of outer for loop
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}](PLA) Error occurred while waiting for Memory response: {e}, retrying in 1 second...")
            time.sleep(1)
    
    # Check response status
    if not memory_response or memory_response.get('status') != 'success':
        logging.warning(f"[{os.getpid()}](PLA) Did not receive valid Memory response or request failed")
        return []
    
    # Extract and return search results
    return memory_response.get('data', [])


def update(redis_conn, text, image=None):
    """
    Send update request to Memory Service and wait for response
    
    Args:
        redis_conn: Redis connection object
        text: Text description to add
        image: PIL image object or None
    
    Returns:
        bool: Whether the operation was successful
    """
    # Encode image
    image_data = encode_image(image) if image else None
    
    # Create request
    request_id = str(uuid.uuid4())
    request = {
        "id": request_id,
        "operation": "update",
        "text": text,
        "image_data": image_data
    }
    
    # Define streams
    requests_stream = STREAMS["memory_requests"]
    responses_stream = STREAMS["memory_responses"]
    
    # Create consumer group (if not exists)
    group_name = f"memory_client_{os.getpid()}"
    try:
        redis_conn.xgroup_create(responses_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PLA) Memory response group already exists: {e}")
        pass
    
    # Send request
    redis_conn.xadd(requests_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](PLA) Sending update request to Memory: {request_id}")
    
    # Wait for response
    memory_response = None
    wait_start_time = time.time()
    max_wait_time = 300  # Maximum wait time in seconds

    while memory_response is None and (time.time() - wait_start_time < max_wait_time):
        try:
            # Use block parameter for efficient waiting
            responses = redis_conn.xreadgroup(
                group_name, f"client_worker_{os.getpid()}", 
                {responses_stream: '>'}, 
                count=20, block=100
            )
            
            # Periodic logging to monitor long waits
            if time.time() - wait_start_time > 30:
                logging.info(f"[{os.getpid()}](PLA) Waiting for Memory response over 30 seconds, request ID: {request_id}")
                wait_start_time = time.time()  # Reset timer to avoid log spam

            if not responses:
                # Block timeout, no messages read, continue to next loop iteration
                continue
            
            for stream, message_list in responses:
                for message_id, data in message_list:
                    try:
                        resp_data = json.loads(data.get('data', '{}'))
                        resp_request_id = resp_data.get('request_id')

                        # Check if this is the expected response
                        if resp_request_id == request_id:
                            # This is the response we're waiting for
                            memory_response = resp_data
                            
                            # Acknowledge message as processed
                            redis_conn.xack(responses_stream, group_name, message_id)
                            
                            logging.info(f"[{os.getpid()}](PLA) Received matching Memory response, request ID: {request_id}, total wait time: {time.time() - wait_start_time:.2f} seconds")
                            
                            # Found response, break out of loop
                            break
                        else:
                            # Not the response we're waiting for, ignore it
                            pass

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}](PLA) Unable to parse or process Memory response message (ID: {message_id}): {e}. Acknowledging this message to prevent infinite loop.")
                        # For unparseable messages, should acknowledge to prevent repeated processing
                        redis_conn.xack(responses_stream, group_name, message_id)
                        continue
                
                if memory_response:
                    break  # Break out of outer for loop
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}](PLA) Error occurred while waiting for Memory response: {e}, retrying in 1 second...")
            time.sleep(1)
    
    # Check response status
    if not memory_response or memory_response.get('status') != 'success':
        logging.warning(f"[{os.getpid()}](PLA) Did not receive valid Memory response or request failed")
        return False
    
    # Operation successful
    return True


def can_stop(redis_conn, question, rgb_im=None, must_stop=False, used_steps=0):
    """
    Send request to Stopping Service asking whether exploration can be stopped
    
    Args:
        redis_conn: Redis connection object
        question: Question object
        images: Image data list, optional
        must_stop: Whether maximum step limit is reached and exploration must stop
        used_steps: Number of steps used, default is 0
    
    Returns:
        dict: Stopping service response, including status and confidence information
    """
    # Encode image
    image_data = encode_image(rgb_im) if rgb_im else None
    
    # Create request
    request_id = str(uuid.uuid4())
    request = {
        "question": question,
        "image": image_data,
        "must_stop": must_stop,
        "used_steps": used_steps
    }
    
    # Define streams
    planner_to_stopping_stream = STREAMS["planner_to_stopping"]
    stopping_to_planner_stream = STREAMS["stopping_to_planner"]
    
    # Create consumer group (if not exists)
    group_name = f"planner_client_{os.getpid()}"
    try:
        redis_conn.xgroup_create(stopping_to_planner_stream, group_name, id='0', mkstream=True)
    except Exception as e:
        # logging.info(f"[{os.getpid()}](PLA) Stopping response group already exists: {e}")
        pass
    
    # Send request
    redis_conn.xadd(planner_to_stopping_stream, {"data": json.dumps(request)})
    logging.info(f"[{os.getpid()}](PLA) Sending request to Stopping Service: {request_id}")
    
    # Wait for response
    stopping_response = None
    wait_start_time = time.time()
    max_wait_time = 300  # Maximum wait time in seconds

    while stopping_response is None and (time.time() - wait_start_time < max_wait_time):
        try:
            # Use block parameter for efficient waiting
            responses = redis_conn.xreadgroup(
                group_name, f"client_worker_{os.getpid()}", 
                {stopping_to_planner_stream: '>'}, 
                count=20, block=100
            )
            
            # Periodic logging to monitor long waits
            if time.time() - wait_start_time > 30:
                logging.info(f"[{os.getpid()}](PLA) Waiting for Stopping Service response over 30 seconds, request ID: {request_id}")
                wait_start_time = time.time()  # Reset timer to avoid log spam

            if not responses:
                # Block timeout, no messages read, continue to next loop iteration
                continue
            
            for stream, message_list in responses:
                for message_id, data in message_list:
                    try:
                        resp_data = json.loads(data.get('data', '{}'))
                        
                        # This is the response we're waiting for
                        stopping_response = resp_data
                        
                        # Acknowledge message as processed
                        redis_conn.xack(stopping_to_planner_stream, group_name, message_id)
                        
                        logging.info(f"[{os.getpid()}](PLA) Received Stopping Service response, total wait time: {time.time() - wait_start_time:.2f} seconds")
                        
                        # Found response, break out of loop
                        break

                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.warning(f"[{os.getpid()}](PLA) Unable to parse or process Stopping response message (ID: {message_id}): {e}. Acknowledging this message to prevent infinite loop.")
                        # For unparseable messages, should acknowledge to prevent repeated processing
                        redis_conn.xack(stopping_to_planner_stream, group_name, message_id)
                        continue
                
                if stopping_response:
                    break  # Break out of outer for loop
        
        except Exception as e:
            logging.warning(f"[{os.getpid()}](PLA) Error occurred while waiting for Stopping response: {e}, retrying in 1 second...")
            time.sleep(1)
    
    # Check response status
    if not stopping_response:
        logging.warning(f"[{os.getpid()}](PLA) Did not receive valid Stopping response")
        # Return a default response indicating to continue exploration
        return {"status": "continue", "confidence": 0.0}
    
    # Return stopping service response
    return stopping_response


def get_group_info(redis_conn, group_id):
    """
    Read information for specified group from Redis
    
    Args:
        redis_conn: Redis connection object
        group_id: Group ID
        
    Returns:
        dict: Group information dictionary
    """
    group_info = {}
    
    # Read basic information (string values stored using get command)
    group_info["group_id"] = redis_conn.get(f"{GROUP_INFO['group_id']}{group_id}")
    group_info["scene"] = redis_conn.get(f"{GROUP_INFO['scene']}{group_id}")
    
    angle = redis_conn.get(f"{GROUP_INFO['angle']}{group_id}")
    if angle:
        group_info["angle"] = float(angle)
    
    # Optional values, read if exists
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
    
    # Read coordinate information (hash values stored using hget command)
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
    
    # Read number of questions
    num_questions_init = redis_conn.get(f"{GROUP_INFO['num_questions_init']}{group_id}")
    if num_questions_init:
        group_info["num_questions_init"] = int(num_questions_init)
    
    num_questions_follow_up = redis_conn.get(f"{GROUP_INFO['num_questions_follow_up']}{group_id}")
    if num_questions_follow_up:
        group_info["num_questions_follow_up"] = int(num_questions_follow_up)
    
    # Read answer mapping
    correct_answers = redis_conn.hgetall(f"{GROUP_INFO['correct_answers']}{group_id}")
    if correct_answers:
        group_info["correct_answers"] = correct_answers
    
    return group_info


def set_group_info(redis_conn, group_id, group_info):
    """
    Write group information to Redis
    
    Args:
        redis_conn: Redis connection object
        group_id: Group ID
        group_info: Group information dictionary
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        pipe = redis_conn.pipeline()
        
        # Set basic information
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
        
        # Set coordinate information
        if "pts" in group_info and isinstance(group_info["pts"], dict):
            pipe.hset(f"{GROUP_INFO['pts']}{group_id}", mapping=group_info["pts"])
        
        if "rotation" in group_info and isinstance(group_info["rotation"], dict):
            pipe.hset(f"{GROUP_INFO['rotation']}{group_id}", mapping=group_info["rotation"])
        elif "rotation" in group_info and isinstance(group_info["rotation"], list):
            # If rotation is a list, convert to dictionary format
            rotation_dict = {str(i): val for i, val in enumerate(group_info["rotation"])}
            pipe.hset(f"{GROUP_INFO['rotation']}{group_id}", mapping=rotation_dict)
        
        # Set number of questions
        if "num_questions_init" in group_info:
            pipe.set(f"{GROUP_INFO['num_questions_init']}{group_id}", group_info["num_questions_init"])
        
        if "num_questions_follow_up" in group_info:
            pipe.set(f"{GROUP_INFO['num_questions_follow_up']}{group_id}", group_info["num_questions_follow_up"])
        
        # Set answer mapping
        if "correct_answers" in group_info and isinstance(group_info["correct_answers"], dict):
            pipe.hset(f"{GROUP_INFO['correct_answers']}{group_id}", mapping=group_info["correct_answers"])
        
        # Execute all commands
        pipe.execute()
        return True
    
    except Exception as e:
        logging.error(f"[{os.getpid()}](PLA) Error occurred while setting group information: {e}")
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
        self.num_workers = self.config.get("planner", {}).get("num_workers", 8)
        
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
        
        # Connect to Redis
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
        # Get current group_id from Redis
        group_id = get_current_group_id(self.redis_conn)
        if not group_id:
            raise ValueError("Unable to find current active group_id")
        
        # Get group information from GROUP_INFO
        group_info = get_group_info(self.redis_conn, group_id)
        
        # Extract question and options from description
        description = question_data.get('description', '')
        
        # Use regular expression to extract question and options
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
            # If unable to match, report error
            raise ValueError(f"Unable to extract question and options from description: {description}")
        
        # Format options as string list
        # choices = str(choices)
        
        # Get answer (A, B, C, D)
        answer = question_data.get('answer', None)
        if answer is None:
            raise ValueError("No answer provided in question data")
        
        # Get scene and floor information from GROUP_INFO
        scene = group_info.get('scene')
        floor = group_info.get('floor', '0')  # Default to 0
        scene_floor = scene + "_" + floor
        
        # Get initial position and angle
        if 'pts' in group_info:
            pts = [
                float(group_info['pts'].get('x', 0)),
                float(group_info['pts'].get('y', 0)),
                float(group_info['pts'].get('z', 0))
            ]
        else:
            # If not found in GROUP_INFO, use default values
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
    
    
    def annotate_object(self, box, cls, rgb_im, depth, cam_pose, room):
        """Process single detected object, including cropping, calling large model, coordinate transformation, etc."""
        # Crop target region
        box_xyxy = box.xyxy[0].cpu()
        x1, y1, x2, y2 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])
        obj_im = rgb_im.crop((x1, y1, x2, y2))
        
        # Call large model for description
        obj_caption = self.vlm_tiny.request_with_retry(image=obj_im, prompt=self.prompt_caption)
        
        # Convert center point to world coordinates
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        world_pos = pixel2world(x, y, depth[int(y), int(x)], cam_pose)
        world_pos = pos_normal_to_habitat(world_pos)
        
        # Return object information
        return {"room": room, "cls": cls, "caption": obj_caption[0], "pos": world_pos.tolist()}


    # Input question_data is similar to the following:
    """
    {'scene': '00797-99ML7CGPqsQ', 'floor': '0', 'question': 'Is the door color darker than the ceiling color?', 'choices': "['Yes', 'No', 'They are the same color', 'The ceiling is darker']", 'question_formatted': 'Is the door color darker than the ceiling color? A) Yes B) No C) They are the same color D) The ceiling is darker. Answer:', 'answer': 'A', 'label': 'Comparison', 'source_image': '00797-99ML7CGPqsQ_0.png'}
    """
    def run(self, question_data, question_ind):
        # Get current group_id from Redis
        group_id = get_current_group_id(self.redis_conn)
        if not group_id:
            raise ValueError("Unable to find current active group_id")
        
        # Get group information from GROUP_INFO
        group_info = get_group_info(self.redis_conn, group_id)
        
        # Before starting exploration, first ask stopping service if question can be answered directly
        stopping_response = can_stop(self.redis_conn, question_data)
        if stopping_response.get("status") == "stop":
            # Can answer question directly, no need for exploration
            logging.info(f"[{os.getpid()}](PLA) Stopping Service decided to answer question directly, confidence: {stopping_response.get('confidence', 0.0)}")
            # Create a basic result object
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
            
            # After answering question directly, no need to update GROUP_INFO
            return result
        
        # Prepare data, start new exploration
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
                
                # Parallel processing of objects
                logging.info(f"[{os.getpid()}](PLA) Starting parallel processing of {len(objects.boxes)} detected objects using {self.num_workers} threads")
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = []
                    for box in objects.boxes:
                        cls = objects.names[box.cls.item()]
                        futures.append(executor.submit(self.annotate_object, box, cls, rgb_im, depth, cam_pose, room))
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        objs_info.append(future.result())
                logging.info(f"[{os.getpid()}](PLA) Parallel processing completed, processed {len(objs_info)} objects in total")

                # "Describe this image."
                caption = self.vlm_lite.request_with_retry(image=rgb_im, prompt=self.prompt_caption)

                if self.config.get("save_obs", True):
                    save_rgbd(rgb, depth, os.path.join(episode_data_dir, f"{cnt_step}_rgbd.png"))
                    rgb_path = os.path.join(episode_data_dir, "{}.png".format(cnt_step))
                    plt.imsave(rgb_path, rgb)
                
                # Build object information
                objs_str = json.dumps(objs_info)
                
                # Add knowledge to Memory
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

                # After each step, ask stopping service whether exploration can be stopped
                # If can be ended, store updated information to GROUP_INFO
                # "... How confident are you in answering this question from your current perspective? ..."
                stopping_response = can_stop(self.redis_conn, question_data, rgb_im, used_steps=cnt_step + 1)
                if stopping_response.get("status") == "stop":
                    # Can stop exploration, end loop
                    logging.info(f"[{os.getpid()}](PLA) Stopping Service decided to stop exploration, confidence: {stopping_response.get('confidence', 0.0)}")
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
            if cnt_step < num_step - 1:  # Avoid calculating next position on the last step
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
            
            # When maximum exploration steps reached, force end exploration
            if cnt_step == num_step - 1:
                logging.info(f"Reached maximum exploration steps {num_step}, forcing end of exploration")
                stopping_response = can_stop(self.redis_conn, question_data, rgb_im, must_stop=True, used_steps=cnt_step + 1)
                if stopping_response.get("status") == "stop":
                    # Can stop exploration, end loop
                    logging.info(f"[{os.getpid()}](PLA) Stopping Service decided to stop exploration, confidence set to 1.0")
                else:
                    raise RuntimeError(
                        f"Reached maximum exploration steps {num_step}, but Stopping Service failed to confirm stopping exploration"
                    )
                
                

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Scene: {scene}, Floor: {floor}")
        logging.info(f"Explored steps: {cnt_step + 1}")
        
        # Record the number of exploration steps
        result["summary"]["explored_steps"] = cnt_step + 1
        
        # Update GROUP_INFO
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
        
        # Only update GROUP_INFO when use_parallel is True
        if self.use_parallel:
            set_group_info(self.redis_conn, group_id, updated_group_info)

        return result
