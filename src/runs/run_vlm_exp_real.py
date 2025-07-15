"""
Run EQA in Real World with VLM exploration.

"""

import os
import json

# os.environ["TRANSFORMERS_VERBOSITY"] = episode_data_dir "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np
np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
import quaternion
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
# from src.habitat import (
#     make_simple_cfg,
#     pos_normal_to_habitat,
#     pos_habitat_to_normal,
#     pose_habitat_to_normal,
#     pose_normal_to_tsdf,
# )
from src.go2 import (
    pos_normal_to_go2,
    pos_go2_to_normal,
    pose_go2_to_normal,
    pose_normal_to_tsdf,
    pose_normal_to_tsdf_real,
    world_to_go2
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from src.utils import interpolate_position_and_rotation, quaternion_to_yaw, move_to_xy, get_vlm_loss, get_delta_yaw


from server_wrapper.go2_flask import Go2


def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    # cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)
    cam_intr = np.array([[908.953     ,   0.        , 656.43      ],
                         [  0.        , 908.915     , 369.506     ],
                         [  0.        ,   0.        ,   1.        ]])

    # Load dataset
    question_data = {"question": "Are there any apples here",
                     "choices": "['There is no relevant information in this scene.', 'Yes, there are apples here.', 'No, there are no apples here.', 'I do not know.']",
                     "answer": "B"}

    # Extract question
    question = question_data["question"]
    choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
    answer = question_data["answer"]

    # Re-format the question to follow LLaMA style
    vlm_question = question
    vlm_pred_candidates = ["A", "B", "C", "D"]
    for token, choice in zip(vlm_pred_candidates, choices):
        vlm_question += "\n" + token + "." + " " + choice
    logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")

    # Set data dir for this question - set initial data to be saved
    episode_data_dir = cfg.output_dir
    os.makedirs(episode_data_dir, exist_ok=True)

    # Init Unitree Go2
    go2 = Go2()
    obs = go2.fetch_rgbd_with_highstate()
    state = obs["highstate"]

    # get pose
    init_pose_data = {"init_pts": np.array([state["position_x"], state["position_y"], state["position_z"]]), "init_angle": quaternion_to_yaw(state["imu_quaternion"])}

    pts = init_pose_data["init_pts"]
    angle = init_pose_data["init_angle"]
    cur_yaw = angle

    next_pts = pts
    next_yaw = cur_yaw

    pts_normal = pts
    floor_height = pts_normal[-1]
    tsdf_bnds = np.array([[-8, 8],
                          [-8, 8],
                          [-0.1, 0.5]]
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
    num_step = 100
    result = {}

    while True:
        cnt_step += 1
        logging.info(f"\n== step: {cnt_step}")

        # Save step info and set current pose
        step_name = f"step_{cnt_step}"
        logging.info(f"Current pts: {pts}, current angle: {cur_yaw}")
        logging.info(f"Next pts: {next_pts}, next angle: {next_yaw}")


        # calculate sport parameters
        next_go2_pts = world_to_go2(np.array(pts[:-1]), np.array(next_pts[:-1]), cur_yaw)
        vx, vy, duration = move_to_xy(np.array([0, 0]), next_go2_pts)
        delta_yaw = get_delta_yaw(cur_yaw, next_yaw)
        
        logging.info(f"vx: {vx}, vy: {vy}, yaw: {delta_yaw}, duration: {duration}")

        # if cnt_step > 0:
        #     import pdb; pdb.set_trace()
        if duration != 0:
            go2.velocity_move(vx, vy, 0, duration)
            time.sleep(duration)
            yaw_duration = go2.turn_vyaw(delta_yaw / np.pi * 4)
            time.sleep(yaw_duration)

        time.sleep(1)
        while True:
            obs = go2.fetch_rgbd_with_highstate() # 这里要获取传感器结果，包括rgb和depth
            rgb = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_BGR2RGB)
            depth = obs["depth_sensor"].astype(np.float32) / 1000 # 毫米转换成米
            state = obs["highstate"]

            if rgb.shape[:2] == depth.shape:
                break
            else:
                print(f"RGB and depth are not aligned. RGB {rgb.shape[:2]}. Depth {depth.shape}")

        # Get observation at current pose - skip black image, meaning robot is outside the floor
        pts = np.array([state["position_x"], state["position_y"], state["position_z"]])
        angle = quaternion_to_yaw(state["imu_quaternion"]) * np.pi

        pts_normal = pts
        result[step_name] = {"pts": pts, "angle": angle}

        # Update camera info
        # wxyz
        quaternion_0 = np.quaternion(state['imu_quaternion'][0], state['imu_quaternion'][1], state['imu_quaternion'][2], state['imu_quaternion'][3])
        translation_0 = pts
        
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0
        cam_pose_normal = cam_pose
        cam_pose_tsdf = pose_normal_to_tsdf_real(cam_pose_normal)

        if cfg.save_obs:
            depth_image = (depth.astype(np.float32) / depth.max()) * 255
            depth_image = np.clip(depth_image, 0, 255).astype(np.uint8)
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
            rgbd = np.concatenate((rgb, depth_image), axis=0)
            plt.imsave(
                os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgbd
            )
        num_black_pixels = np.sum(
            np.sum(rgb, axis=-1) == 0
        )  # sum over channel first
        if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:
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
            tsdf_planner.get_mesh("results/scenes/419.ply")

            # Get frontier candidates
            prompt_points_pix = []
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
                logging.info(f"prompt points: {prompt_points_pix}")
                fig.tight_layout()
                # plt.savefig(
                #     os.path.join(
                #         episode_data_dir, "{}_prompt_points.png".format(cnt_step)
                #     )
                # )
                print("save prompt points.")
                plt.savefig(
                    os.path.join(
                        episode_data_dir, "prompt_points.png".format(cnt_step)
                    )
                )
                plt.close()

            # Get VLM prediction
            rgb_im = Image.fromarray(rgb)
            prompt_question = (
                vlm_question
                + "\nAnswer with the option's letter from the given choices directly."
            )
            smx_vlm_pred = get_vlm_loss(rgb_im, prompt_question, vlm_pred_candidates)
            logging.info(f"Pred - Prob: {smx_vlm_pred}")

            # Get VLM relevancy
            prompt_rel = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view? Answer with Yes or No."
            smx_vlm_rel = get_vlm_loss(rgb_im, prompt_rel, ["Yes", "No"])
            logging.info(f"Rel - Prob: {smx_vlm_rel}")

            # Visual prompting
            draw_letters = ["A", "B", "C", "D"]  # always four
            fnt = ImageFont.truetype(
                "data/Open_Sans/static/OpenSans-Regular.ttf",
                30,
            )
            actual_num_prompt_points = len(prompt_points_pix)
            if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
                rgb_im_draw = rgb_im.copy()
                draw = ImageDraw.Draw(rgb_im_draw)
                for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
                    draw.ellipse(
                        (
                            point_pix[0] - cfg.visual_prompt.circle_radius,
                            point_pix[1] - cfg.visual_prompt.circle_radius,
                            point_pix[0] + cfg.visual_prompt.circle_radius,
                            point_pix[1] + cfg.visual_prompt.circle_radius,
                        ),
                        fill=(200, 200, 200, 255),
                        outline=(0, 0, 0, 255),
                        width=3,
                    )
                    draw.text(
                        tuple(point_pix.astype(int).tolist()),
                        draw_letters[prompt_point_ind],
                        font=fnt,
                        fill=(0, 0, 0, 255),
                        anchor="mm",
                        font_size=12,
                    )
                rgb_im_draw.save(
                    os.path.join(episode_data_dir, f"{cnt_step}_draw.png")
                )

                # get VLM reasoning for exploring
                if cfg.use_lsv:
                    prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Answer with a single letter."
                    lsv = get_vlm_loss(rgb_im_draw, prompt_lsv, draw_letters[:actual_num_prompt_points])
                    lsv *= actual_num_prompt_points / 3
                else:
                    lsv = (
                        np.ones(actual_num_prompt_points) / actual_num_prompt_points
                    )

                # base - use image without label
                if cfg.use_gsv:
                    prompt_gsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                    gsv = get_vlm_loss(rgb_im, prompt_gsv, ["Yes", "No"])[0]
                    gsv = (
                        np.exp(gsv / cfg.gsv_T) / cfg.gsv_F
                    )  # scalenum_step before combined with lsv
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

            # Save data
            result[step_name]["smx_vlm_pred"] = smx_vlm_pred
            result[step_name]["smx_vlm_rel"] = smx_vlm_rel
        else:
            logging.info("Skipping black image!")
            result[step_name]["smx_vlm_pred"] = (np.ones((4)) / 4)
            result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])

        # Determine next point
        if cnt_step < num_step:
            pts_normal, next_yaw, cur_yaw, pts_pix, fig = tsdf_planner.find_next_pose(
                pts=pts_normal,
                angle=angle,
                cam_pose=cam_pose_tsdf,
                flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                **cfg.planner,
            )
            pts_pixs = np.vstack((pts_pixs, pts_pix))
            pts_normal = np.append(pts_normal, floor_height)
            next_pts = pts_normal

            # Add path to ax5, with colormap to indicate order
            ax5 = fig.axes[4]
            ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
            ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
            fig.tight_layout()
            # plt.savefig(
            #     os.path.join(episode_data_dir, "{}_map.png".format(cnt_step + 1))
            # )
            plt.savefig(
                os.path.join(episode_data_dir, "map.png".format(cnt_step + 1))
            )
            plt.close()

        candidates = ["A", "B", "C", "D"]
        choice_idx = np.argmax(smx_vlm_pred)
        pred_token = candidates[choice_idx]
        success_weighted = pred_token == answer
        result[step_name]['answer_weighted'] = choices[choice_idx]

        smx_vlm = (smx_vlm_rel[0] * smx_vlm_pred)
        choice_idx = np.argmax(smx_vlm)
        pred_token = candidates[choice_idx]
        success_max = pred_token == answer
        result[step_name]['answer_max'] = choices[choice_idx]

        if success_max or success_weighted:
            break

    # Check if success using weighted prediction
    smx_vlm_all = np.empty((0, 4))
    relevancy_all = []
    candidates = ["A", "B", "C", "D"]
    for step in range(cnt_step + 1):
        smx_vlm_pred = result[f"step_{step}"]["smx_vlm_pred"]
        smx_vlm_rel = result[f"step_{step}"]["smx_vlm_rel"]
        relevancy_all.append(smx_vlm_rel[0])
        smx_vlm_all = np.vstack((smx_vlm_all, smx_vlm_rel[0] * smx_vlm_pred))

    # Option 1: use the max of the weighted predictions
    smx_vlm_max = np.max(smx_vlm_all, axis=0)
    pred_token = candidates[np.argmax(smx_vlm_max)]
    success_weighted = pred_token == answer
    # Option 2: use the max of the relevancy
    max_relevancy = np.argmax(relevancy_all)
    relevancy_ord = np.flip(np.argsort(relevancy_all))
    pred_token = candidates[np.argmax(smx_vlm_all[max_relevancy])]
    success_max = pred_token == answer

    # Episode summary
    logging.info(f"\n== Episode Summary")
    logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")
    logging.info(f"Success (weighted): {success_weighted}")
    logging.info(f"Success (max): {success_max}")
    logging.info(
        f"Top 3 steps with highest relevancy with value: {relevancy_ord[:3]} {[relevancy_all[i] for i in relevancy_ord[:3]]}"
    )
    logging.info(f"\n== All Summary")
    logging.info(f"Number of data collected: {cnt_step}")
    logging.info(f"Answer (weighted): {result[step_name]['answer_weighted']}")
    logging.info(f"Answer (max): {result[step_name]['answer_max']}")


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
