# utils/utils.py

import os
import numpy as np
import math
import io
import requests
import json
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Slerp, Rotation as R
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def interpolate_position_and_rotation(points, start_rot, end_rot, num_intermediate_points=10):
    # Initialize result path
    dense_path = []
    
    # Convert start and end rotations to scipy Rotation objects for interpolation
    start_rot = np.array([start_rot.x, start_rot.y, start_rot.z, start_rot.w])
    end_rot = np.array(end_rot)
    rotations = R.from_quat([start_rot, end_rot])
    slerp = Slerp([0, len(points) - 1], rotations)

    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        dense_path.append((start, slerp(i).as_quat()))  # Add start point and its rotation
        for j in range(1, num_intermediate_points + 1):
            # Position interpolation
            interpolated_position = start + (end - start) * (j / (num_intermediate_points + 1))
            # Rotation interpolation
            interpolated_rotation = slerp(i + j / (num_intermediate_points + 1)).as_quat()
            dense_path.append((interpolated_position, interpolated_rotation))
    dense_path.append((points[-1], end_rot))  # Add end point and its rotation
    return dense_path


def quaternion_to_yaw(quat):
    # Calculate yaw angle
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (x**2 + y**2))
    return yaw


def pts_to_distance(pts, dst_pts):
    return np.linalg.norm(dst_pts - pts)


def calculate_angle(a, b):
    # Calculate dot product
    dot_product = np.dot(a, b)
    # Calculate vector magnitude
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Calculate cosine value
    cos_theta = max(-1, min(1, dot_product / (norm_a * norm_b)))
    # Calculate angle through arccosine function (unit: radians)
    angle_rad = np.arccos(cos_theta)
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_rad, angle_deg


def move_to_xy(pts, dst_pts):
    distance = pts_to_distance(pts, dst_pts)
    yaw, angle = calculate_angle(pts, dst_pts)
    return distance, (yaw, angle)


def move_to_xy_with_yaw(pts, dst_pts, yaw, dst_yaw, vx=0.3):
    vector = dst_pts - pts
    norm_vector = np.linalg.norm(vector)

    tanx = vector[1] / vector[0]
    cosx = vector[1] / norm_vector

    dx = norm_vector * cosx
    delta_yaw = (dst_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

    vy = tanx * vx
    duration = dx / vy
    vyaw = delta_yaw / duration
    return vx, vy, vyaw, duration


def get_delta_yaw(yaw, dst_yaw):
    return (dst_yaw - yaw + math.pi) % (2 * math.pi) - math.pi


def move_to_xy(pts, dst_pts, v=0.3):
    
    dx = abs(dst_pts[0] - pts[0])
    dy = abs(dst_pts[1] - pts[1])

    if dx == 0.0 and dx == dy:
        return 0.0, 0.0, 0.0
    
    if dx > dy:
        duration = dx / v
    else:
        duration = dy / v

    vx = dx / duration
    vy = dy / duration

    if dst_pts[0] - pts[0] < 0:
        vx = -vx
    if dst_pts[1] - pts[1] < 0:
        vy = -vy

    return vx, vy, duration


def display_sample(rgb, depth, save_path="sample.png"):
    # Create subplots with 3 columns
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))

    # Display RGB image
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')  # Turn off coordinate axes

    # Display depth image
    axes[1].imshow(depth, cmap='jet')  # Use 'jet' color scheme
    axes[1].set_title("Depth Image")
    axes[1].axis('off')  # Turn off coordinate axes

    # Adjust subplot layout
    plt.tight_layout()

    # Save image as PNG file
    plt.savefig(save_path, format="png")

    plt.close()


def draw_letters(rgb_im, prompt_points_pix, letters, circle_radius, fnt, save_path):
    rgb_im_draw = rgb_im.copy()
    draw = ImageDraw.Draw(rgb_im_draw)
    for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
        draw.ellipse(
            (
                point_pix[0] - circle_radius,
                point_pix[1] - circle_radius,
                point_pix[0] + circle_radius,
                point_pix[1] + circle_radius,
            ),
            fill=(200, 200, 200, 255),
            outline=(0, 0, 0, 255),
            width=3,
        )
        draw.text(
            tuple(point_pix.astype(int).tolist()),
            letters[prompt_point_ind],
            font=fnt,
            fill=(0, 0, 0, 255),
            anchor="mm",
            font_size=12,
        )
    rgb_im_draw.save(save_path)
    return rgb_im_draw


def save_rgbd(rgb, depth, save_path="rgbd.png"):
    depth_image = (depth.astype(np.float32) / depth.max()) * 255
    depth_image = np.clip(depth_image, 0, 255).astype(np.uint8)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGRA)

    rgbd = np.concatenate((rgb, depth_image), axis=0)
    plt.imsave(save_path, rgbd)


def pixel2world(x, y, depth, pose):
    pos = np.array([x, y, depth])
    pos = np.dot(np.linalg.inv(pose[:3, :3]), pos - pose[:3, 3])
    pos = np.dot(pose[:3, :3], pos) + pose[:3, 3]
    return pos
