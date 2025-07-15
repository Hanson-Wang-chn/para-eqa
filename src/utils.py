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
    # 初始化结果路径
    dense_path = []
    
    # 转换起点和终点旋转为 scipy 的 Rotation 对象以便插值
    start_rot = np.array([start_rot.x, start_rot.y, start_rot.z, start_rot.w])
    end_rot = np.array(end_rot)
    rotations = R.from_quat([start_rot, end_rot])
    slerp = Slerp([0, len(points) - 1], rotations)

    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        dense_path.append((start, slerp(i).as_quat()))  # 添加起点及其旋转
        for j in range(1, num_intermediate_points + 1):
            # 位置插值
            interpolated_position = start + (end - start) * (j / (num_intermediate_points + 1))
            # 旋转插值
            interpolated_rotation = slerp(i + j / (num_intermediate_points + 1)).as_quat()
            dense_path.append((interpolated_position, interpolated_rotation))
    dense_path.append((points[-1], end_rot))  # 添加终点及其旋转
    return dense_path

def quaternion_to_yaw(quat):
    # 计算 yaw 角度
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (x**2 + y**2))
    return yaw

def pts_to_distance(pts, dst_pts):
    return np.linalg.norm(dst_pts - pts)

def get_vlm_loss(image, prompt, tokens):
    # 调用这个方法需要先启动vlm服务
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")  # 可以指定图片格式
    img_byte_arr.seek(0)

    files = {'image': img_byte_arr}
    data = {
        "text": prompt,
        "str_list": json.dumps(tokens)
    }

    response = requests.post("http://127.0.0.1:5000/get_loss", files=files, data=data)
    
    result = json.loads(response.text)
    return np.array(result['result'])

def get_vlm_response(image, prompt, kb=None):
    # 调用这个方法需要先启动vlm服务
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")  # 可以指定图片格式
    img_byte_arr.seek(0)

    kb = json.dumps(kb) if kb is not None else json.dumps([])
    files = {'image': img_byte_arr}
    data = {
        "text": prompt,
        "kb": kb
    }

    response = requests.post("http://127.0.0.1:5000/get_response", files=files, data=data)

    result = json.loads(response.text)
    return np.array(result['result'])

def calculate_angle(a, b):
    # 计算点积
    dot_product = np.dot(a, b)
    # 计算向量模
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # 计算余弦值
    cos_theta = max(-1, min(1, dot_product / (norm_a * norm_b)))
    # 通过反余弦函数求夹角（单位：弧度）
    angle_rad = np.arccos(cos_theta)
    # 转换为角度
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