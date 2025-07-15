"""
Run EQA in Habitat-Sim with VLM exploration.

"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np
import time
np.set_printoptions(precision=3)
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
import random
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from src.utils import interpolate_position_and_rotation, get_vlm_loss

import matplotlib.pyplot as plt

def display_sample(rgb, depth, save_path="sample.png"):
    # 创建一个包含3列的子图
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))

    # 显示RGB图像
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')  # 关闭坐标轴

    # 显示深度图像
    axes[1].imshow(depth, cmap='jet')  # 使用 'jet' 配色方案
    axes[1].set_title("Depth Image")
    axes[1].axis('off')  # 关闭坐标轴

    # 调整子图布局
    plt.tight_layout()

    # 保存图像为PNG文件
    plt.savefig(save_path, format="png")

    plt.close()

# 计算四元数的逆
def quat_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# 计算四元数的角度（单位：弧度）
def quat_angle(q1, q2):
    q_rel = quat_multiply(q2, quat_inverse(q1))  # 计算 q2 到 q1 的相对旋转
    angle = 2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))  # 计算旋转角度
    return angle, q_rel[0]  # 返回角度和实部（用于判断旋转方向）

# 四元数乘法
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

# 判断需要旋转多少次，每次旋转 30 度，并返回旋转方向
def rotation_steps(A, B, step_angle_deg=30):
    step_angle_rad = np.radians(step_angle_deg)  # 转换为弧度
    angle, real_part = quat_angle(A, B)  # 计算从 A 到 B 的旋转角度
    if angle > np.pi:  # 如果角度大于 180 度
        angle = 2 * np.pi - angle  # 选择最短旋转方向
    
    # 判断旋转方向
    direction = "counterclockwise" if real_part >= 0 else "clockwise"
    
    steps = np.ceil(angle / step_angle_rad)  # 计算需要的旋转次数
    return int(steps), direction



def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    logging.info(f"Loaded {len(questions_data)} questions.")

    # Run all questions
    cnt_data = 0
    results_all = []
    for question_ind in tqdm(range(len(questions_data))):

        # Extract question
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]
        logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")

        # Re-format the question to follow LLaMA style
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]
        for token, choice in zip(vlm_pred_candidates, choices):
            vlm_question += "\n" + token + "." + " " + choice
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.output_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)
        result = {"question_ind": question_ind}

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        scene_mesh_dir = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".glb"
        )
        navmesh_file = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
        )
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_file)
        if not pathfinder.is_loaded:
            print("Not loaded .navmesh file yet. Please check file path {}.".format(navmesh_file))

        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = init_pts
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
        ).tolist()
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )

        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pts_normal,
            init_clearance=cfg.init_clearance * 2,
        )

        # Run steps
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        cnt_step = -1
        while True:
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            
            # 第二步开始将位置转换为action
            if cnt_step > 0:
                goal_position = pts
                if pathfinder.is_navigable(goal_position):
                    path = habitat_sim.ShortestPath()
                    path.requested_start = agent.get_state().sensor_states["depth_sensor"].position
                    path.requested_end = goal_position

                    # 如果路径找到，使用GreedyFollower生成动作序列
                    follower = habitat_sim.GreedyGeodesicFollower(
                        pathfinder, agent, goal_radius=0.1
                    )
                    # 按路径生成动作
                    actions = []
                    idx = -1
                    # 位移
                    while True:
                        try:
                            idx += 1
                            action = follower.next_action_along(goal_position)

                            if action is None:
                                break

                            actions.append(action)

                            # 模拟执行动作
                            if action == "move_forward":
                                observations = simulator.step("move_forward")
                            elif action == "turn_left":
                                observations = simulator.step("turn_left")
                            elif action == "turn_right":
                                observations = simulator.step("turn_right")

                            print(agent.get_state().sensor_states["depth_sensor"].position, "--->", goal_position)
                            if cfg.save_obs:
                                rgb = observations["color_sensor"]
                                depth = observations["depth_sensor"]
                                time.sleep(0.5)
                                display_sample(rgb, depth, 'results/test_scene/sample.png')

                            pts_normal = pos_habitat_to_normal(agent.get_state().sensor_states["depth_sensor"].position)

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
                            obs = simulator.get_sensor_observations()
                            rgb = obs["color_sensor"]
                            depth = obs["depth_sensor"]

                            if cfg.save_obs:
                                display_sample(rgb, depth, "results/test_scene/sample.png")

                            # TSDF fusion
                            tsdf_planner.integrate(
                                color_im=rgb,
                                depth_im=depth,
                                cam_intr=cam_intr,
                                cam_pose=cam_pose_tsdf,
                                obs_weight=1.0,
                                margin_h=int(cfg.margin_h_ratio * img_height),
                                margin_w=int(cfg.margin_w_ratio * img_width),
                            )
                            tsdf_planner.get_mesh(f"results/test_scene/scene_{cnt_data}.ply")

                            if cfg.use_active:
                                prompt_points_pix, fig = (
                                    tsdf_planner.find_prompt_points_within_view(
                                        pts_normal,
                                        img_width,
                                        img_height,
                                        cam_intr,
                                        cam_pose_tsdf,
                                        **cfg.visual_prompt,
                                    )
                                )
                                fig.tight_layout()
                                plt.savefig("results/test_scene/prompt_points.png")
                                plt.close()

                        except habitat_sim.errors.GreedyFollowerError as e:
                            print("GreedyFollowerError encountered:", e)
                            break
                    # 旋转
                    rot = agent.get_state().sensor_states["depth_sensor"].rotation.tolist()
                    rot = [rot.w, rot.x, rot.y, rot.x]
                    steps, direction = rotation_steps(rot, rotation)
                    for _ in range(steps):
                        if direction == "counterclockwise":
                            observations = simulator.step("turn_left")
                        elif direction == "clockwise":
                            observations = simulator.step("turn_right")
                        if cfg.save_obs:
                            rgb = observations["color_sensor"]
                            depth = observations["depth_sensor"]
                            time.sleep(0.5)
                            display_sample(rgb, depth, 'results/test_scene/sample.png')

                    print("Actions to reach goal:", actions)
                else:
                    print("目标位置不可达")

            agent_state.position = pts
            agent_state.rotation = rotation
            agent.set_state(agent_state)

            # pts_normal = pos_habitat_to_normal(pts)
            # result[step_name] = {"pts": pts, "angle": angle}

            # # Update camera info
            # sensor = agent.get_state().sensor_states["depth_sensor"]
            # quaternion_0 = sensor.rotation
            # translation_0 = sensor.position

            # cam_pose = np.eye(4)
            # cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
            # cam_pose[:3, 3] = translation_0
            # cam_pose_normal = pose_habitat_to_normal(cam_pose)
            # cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

            # # Get observation at current pose - skip black image, meaning robot is outside the floor
            # obs = simulator.get_sensor_observations()
            # rgb = obs["color_sensor"]
            # depth = obs["depth_sensor"]

            # if cfg.save_obs:
            #     display_sample(rgb, depth, "results/test_scene/sample.png")

            # # TSDF fusion
            # tsdf_planner.integrate(
            #     color_im=rgb,
            #     depth_im=depth,
            #     cam_intr=cam_intr,
            #     cam_pose=cam_pose_tsdf,
            #     obs_weight=1.0,
            #     margin_h=int(cfg.margin_h_ratio * img_height),
            #     margin_w=int(cfg.margin_w_ratio * img_width),
            # )
            # tsdf_planner.get_mesh(f"results/test_scene/scene_{cnt_data}.ply")

            # if cfg.use_active:
            #     prompt_points_pix, fig = (
            #         tsdf_planner.find_prompt_points_within_view(
            #             pts_normal,
            #             img_width,
            #             img_height,
            #             cam_intr,
            #             cam_pose_tsdf,
            #             **cfg.visual_prompt,
            #         )
            #     )
            #     fig.tight_layout()
            #     plt.savefig("results/test_scene/prompt_points.png")
            #     plt.close()

            pts_normal, angle, pts_pix, fig = tsdf_planner.find_next_pose(
                pts=pts_normal,
                angle=angle,
                flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                **cfg.planner,
            )
            pts_pixs = np.vstack((pts_pixs, pts_pix))
            pts_normal = np.append(pts_normal, floor_height)
            pts = pos_normal_to_habitat(pts_normal)

            # Add path to ax5, with colormap to indicate order
            ax5 = fig.axes[4]
            ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
            ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
            fig.tight_layout()
            plt.savefig("results/test_scene/map.png")
            plt.close()

            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
            ).tolist()

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
