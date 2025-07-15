import numpy as np
import logging
import requests
import yaml
import cv2

from server_wrapper.server_wrapper import send_request, ServerMixin, host_model, str_to_image, string_to_numpy, json_to_pointcloud2
# from server_wrapper import send_request, ServerMixin, host_model, str_to_image, string_to_numpy, json_to_pointcloud2

import json



def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    Convert a quaternion to a 4x4 transformation matrix.
    """
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w

    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - zw)
    r02 = 2 * (xz + yw)
    r10 = 2 * (xy + zw)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - xw)
    r20 = 2 * (xz - yw)
    r21 = 2 * (yz + xw)
    r22 = 1 - 2 * (xx + yy)

    matrix = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])

    return matrix


class Go2():
    def __init__(self, **kwargs):
        self.publish_url = kwargs.get('publish_url', 'http://10.24.6.32:10401/robot_server')
        self.subscribe_url = kwargs.get('subscribe_url', 'http://10.24.6.32:10305/')

        self.frame_number = 0

    def move_forward_vx(self, vx):
        self.move(vx, 0.0, 0.0)

    def turn(self, vyaw):
        self.move(0.0, 0.0, vyaw)

    def move_forward(self, vx, duration):
        self.velocity_move(vx, 0.0, 0.0, duration)

    def move_forward(self, distance):
        if distance > 0:
            vx = 0.5
        else:
            vx = -0.5
        duration = distance / vx
        self.velocity_move(vx, 0.0, 0.0, duration)

    def turn_vyaw(self, vyaw, duration):
        # 正数左转，负数右转
        self.velocity_move(0.0, 0.0, vyaw, duration)

    def turn_vyaw(self, yaw):
        # 正数左转，负数右转
        if yaw > 0:
            vyaw = 0.5
        else:
            vyaw = -0.5
        duration = yaw / vyaw
        self.velocity_move(0.0, 0.0, vyaw, duration)
        return duration

    def move(self, vx, vy, vyaw):
        return send_request(self.publish_url, cmd_id="move",
                            cmd_value={"linear_x": vx, "linear_y": vy, "angular_z": vyaw})

    def velocity_move(self, vx, vy, vyaw, duration):
        return send_request(self.publish_url, cmd_id="velocity_move",
                            cmd_value={"linear_x": vx, "linear_y": vy, "angular_z": vyaw, "duration": duration})

    def euler(self, roll, pitch, yaw):
        return send_request(self.publish_url, cmd_id="euler",
                            cmd_value={"roll": roll, "pitch": pitch, "yaw": yaw})

    def execute_action(self, action_name):
        # "Lay Down": self.sport_client.StandDown, # 1005
        # "Stand Up": self.sport_client.RecoveryStand, # 1006

        # "Sit": self.sport_client.Sit, # 1009
        # "RiseSit": self.sport_client.RiseSit, #1010
        # "Hello" : self.sport_client.Hello, # 1016
        # "Stretch": self.sport_client.Stretch, # 1017
        # "Wallow": self.sport_client.Wallow, # 1021
        # "Dance1": self.sport_client.Dance1, # 1022
        # "Dance2": self.sport_client.Dance2, # 1023

        # "Scrape": self.sport_client.Scrape, # 1029
        # "FrontFlip": self.sport_client.FrontFlip, # 1030
        # "FrontJump": self.sport_client.FrontJump, # 1031
        # "FrontPounce": self.sport_client.FrontPounce, # 1032
        # "WiggleHips": self.sport_client.WiggleHips, # 1033
        # "Heart": self.sport_client.Heart, # 1036

        # #below API only works while in advanced mode
        # "HandStand": self.sport_client.HandStand, #1301
        return send_request(self.publish_url, cmd_id="command", cmd_value=action_name)

    def switch_gait(self, mode_cmd):
        mode_cmd_2_num = {
            "idle": 0,
            "trot": 1,
            "trot running": 2,
            "climb_forward": 3,
            "climb_backward": 4,
        }
        mode_num = mode_cmd_2_num[mode_cmd]

        return send_request(self.publish_url, cmd_id="switch_gait", cmd_value=mode_num)

    def play_sound(self, sound_name):
        return send_request(self.publish_url, cmd_id="play_sound", cmd_value=sound_name)

    # def follow_trajectory(self, trajectory):
    #     return send_request(self.publish_url, cmd_id="trajectory_follow", cmd_value=trajectory)

    # Uncomment if you want to use these methods
    # def get_image(self):
    #     image = send_request(self.subscribe_url, camera_id='front')['image']
    #     return str_to_image(image)

    # def get_pointcloud(self):
    #     pointcloud = send_request(self.subscribe_url, cloud_id='radar')
    #     data = json_to_pointcloud2(pointcloud['pointcloud'])
    #     return data

    # def get_transform(self):
    #     extrinsic = send_request(self.subscribe_url, tf_id='base_link')
    #     data = json.loads(extrinsic["extrinsic"])

    #     extrinsic_all = data["transforms"]
    #     for i in extrinsic_all:
    #         if i["child_frame_id"] == "base_link":
    #             extrinsic_quaternion = i["transform"]
    #             x, y, z, w = extrinsic_quaternion["translation"]["x"], extrinsic_quaternion["translation"]["y"], extrinsic_quaternion["translation"]["z"], extrinsic_quaternion["rotation"]["w"]
    #             extrinsic = quaternion_to_matrix(x, y, z, w)
    #             break
    #     return extrinsic

    # def get_xy_yaw(self):
    #     extrinsic = self.get_transform()
    #     x, y, yaw = extrinsic[0, 3], extrinsic[1, 3], np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    #     return x, y, yaw

    def fetch_video_with_highstate(self, save_data=False):
        highstate = None
        frame = None
        frame_number = self.frame_number
        url = self.subscribe_url + "yield_video_with_highstate"

        # print(f"Connecting to {url}")
        stream = requests.get(url, stream=True, timeout=10)
        # print("Connection established")

        bytes_data = b''
        json_buffer = b''  # 用于积累 JSON 数据

        for chunk in stream.iter_content(chunk_size=1024):
            if not chunk:
                continue
            bytes_data += chunk

            # 调试输出每个块的长度
            # print(f"Received chunk of size {len(chunk)}")

            # 寻找JPEG帧
            start_img = bytes_data.find(b'\xff\xd8')
            end_img = bytes_data.find(b'\xff\xd9')
            end_of_data = bytes_data.find(b'--end\r\n')

            if start_img != -1 and end_img != -1:
                jpg = bytes_data[start_img: end_img + 2]
                bytes_data = bytes_data[end_img + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

                if frame is not None and save_data:
                    frame_filename = f"images/frames/{frame_number:08d}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    print(f"Saved {frame_filename}")
                    self.frame_number += 1

            # 处理JSON数据，确保积累到完整的 JSON
            json_start = bytes_data.find(b'Content-Type: application/json\r\n\r\n')
            if json_start != -1:
                # 提取出 JSON 数据的部分，直到 "end" 标记
                json_data_start = json_start + len(b'Content-Type: application/json\r\n\r\n')
                end_json = bytes_data.find(b"\r\n--end\r\n")

                # 如果找到 JSON 的结尾，提取并解析
                if end_json != -1:
                    json_buffer = bytes_data[json_data_start:end_json]
                    # print(f"Complete JSON data: {json_buffer}")

                    # 解析 JSON 数据
                    highstate = json.loads(json_buffer.decode('utf-8'))
                    print("Received highstate:", highstate)
                    # 清空已处理的数据
                    bytes_data = bytes_data[end_json + len(b"\r\n--end\r\n"):]

                    if highstate is not None and save_data:
                        json_filename = f"images/json/{frame_number:08d}.json"
                        with open(json_filename, 'w') as f:
                            json.dump(highstate, f, indent=4)

            if end_of_data != -1:
                break

        return {"frame": frame, "highstate": highstate}

    def fetch_rgbd_with_highstate(self, save_data=False):
        highstate = None
        frame = None
        frame_number = self.frame_number
        url = self.subscribe_url + "yield_rgbd_with_highstate"
        stream = requests.get(url, stream=True, timeout=10)

        bytes_data = b''
        json_buffer = b''  # 用于积累 JSON 数据

        for chunk in stream.iter_content(chunk_size=1024):
            if not chunk:
                continue
            bytes_data += chunk

            # 调试输出每个块的长度
            # print(f"Received chunk of size {len(chunk)}")

            # 寻找JPEG帧
            start_img = bytes_data.find(b'Content-Type: rgb\r\n\r\n') + len(b'Content-Type: rgb\r\n\r\n')
            end_img = bytes_data.find(b'--rgb_end')

            start_depth = bytes_data.find(b'Content-Type: depth\r\n\r\n') + len(b'Content-Type: depth\r\n\r\n')
            end_depth = bytes_data.find(b'--depth_end')

            end_of_data = bytes_data.find(b'--end\r\n')

            # 处理 RGB 图像
            if start_img != -1 and end_img != -1:
                png_data = bytes_data[start_img: end_img + 2]
                bytes_data = bytes_data[end_img + 2:]
                rgb = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_COLOR)

                # if frame is not None and save_data:
                #     frame_filename = f"/home/unitree/smbu_robot/go2_dashboard/images/images/frames/rgb/{frame_number:08d}.png"  # 保存为 .png
                #     cv2.imwrite(frame_filename, frame)  # 保存 .png 文件
                #     print(f"Saved {frame_filename}")

            # 处理深度图像
            if start_depth != -1 and end_depth != -1:
                depth_data = bytes_data[start_depth: end_depth + 2]
                bytes_data = bytes_data[end_depth + 2:]

                # 将深度数据解码为原始格式 (保持16位深度信息)
                depth = cv2.imdecode(np.frombuffer(depth_data, np.uint8), cv2.IMREAD_UNCHANGED)

                # if frame is not None and save_data:
                #     depth_filename = f"/home/unitree/smbu_robot/go2_dashboard/images/images/frames/depth/{frame_number:08d}.png"  # 保存为 .png
                #     cv2.imwrite(depth_filename, frame)  # 保存 .png 文件，保留16位深度信息
                #     print(f"Saved {depth_filename}")
                #     self.frame_number += 1

            json_start = bytes_data.find(b'Content-Type: application/json\r\n\r\n')
            if json_start != -1:
                # 提取出 JSON 数据的部分，直到 "end" 标记
                json_data_start = json_start + len(b'Content-Type: application/json\r\n\r\n')
                end_json = bytes_data.find(b"\r\n--end\r\n")

                # 如果找到 JSON 的结尾，提取并解析
                if end_json != -1:
                    json_buffer = bytes_data[json_data_start:end_json]

                    # 解析 JSON 数据
                    highstate = json.loads(json_buffer.decode('utf-8'))

                    # 提取四元数数据和平移数据
                    if 'imu_quaternion' in highstate and 'position' in highstate:
                        quaternion = highstate['imu_quaternion']
                        position = highstate['position']  # 平移数据
                        if len(quaternion) == 4 and len(position) == 3:
                            w, x, y, z = quaternion

                            # 将四元数转换为3x3旋转矩阵
                            rotation_matrix = quaternion_to_matrix(x, y, z, w)

                            # 构造齐次变换矩阵
                            transform_matrix = np.eye(4)
                            transform_matrix[:3, :3] = rotation_matrix  # 设置旋转部分
                            transform_matrix[0, 3] = position[1]  # 设置平移 x
                            transform_matrix[1, 3] = position[2]  # 设置平移 y
                            transform_matrix[2, 3] = position[0]  # 设置平移 z

                            # # 直接将矩阵写入txt文件
                            # with open(f'/home/unitree/smbu_robot/go2_dashboard/images/images/json/pose/{frame_number:08d}.txt', 'w') as f:
                            #     for row in transform_matrix[:3]:
                            #         f.write("\t".join(map(str, row)) + "\n")
                            #     # 最后一行是[0.0, 0.0, 0.0, 1.0]
                            #     f.write("0.0\t0.0\t0.0\t1.0\n")

                            # print(f"Saved pose to /home/unitree/smbu_robot/go2_dashboard/images/images/json/pose/{frame_number:08d}.txt")

                    # 清空已处理的数据
                    bytes_data = bytes_data[end_json + len(b"\r\n--end\r\n"):]

                    # if highstate is not None and save_data:
                    #     json_filename = f"/home/unitree/smbu_robot/go2_dashboard/images/images/json/{frame_number:08d}.json"
                    #     with open(json_filename, 'w') as f:
                    #         json.dump(highstate, f, indent=4)

            if end_of_data != -1:
                break

        return {"color_sensor": rgb, "depth_sensor": depth, "highstate": highstate}

    def fetch_height_map_data_with_highstate(self, save_data=False):
        highstate = None
        height_map = None
        height_map_height = 128
        height_map_width = 128
        url = self.subscribe_url + "yield_height_map_data"

        stream = requests.get(url, stream=True, timeout=10)

        bytes_data = b''  # 用于积累数据
        json_buffer = b''  # 用于积累 JSON 数据

        for chunk in stream.iter_content(chunk_size=1024):
            if not chunk:
                continue
            bytes_data += chunk

            # 寻找高度图数据
            height_map_end_index = bytes_data.find(b'Content-Type: height_map\r\n\r\n')
            end_of_data = bytes_data.find(b'--end\r\n')

            if height_map_end_index != -1:
                height_map_data_start = height_map_end_index + len(b'Content-Type: height_map\r\n\r\n')
                height_map_data_end = bytes_data.find(b'--heigh_map_end\r\n', height_map_data_start)
                if height_map_data_end != -1:
                    height_map_data = bytes_data[height_map_data_start:height_map_data_end]
                    bytes_data = bytes_data[height_map_data_end + len(b'--heigh_map_end\r\n'):]

                    height_map_data = json.loads(height_map_data.decode('utf-8'))

                    # # 转换为 np.array，假设高度图数据为浮点数格式
                    height_map_array = np.array(height_map_data['data'], dtype=np.float32)
                    height_map_array = height_map_array.reshape((height_map_height, height_map_width))
                    height_map_data['data'] = height_map_array
                    # height_map_array = np.frombuffer(height_map_data, dtype=np.float32)  # 根据具体数据格式调整 dtype
                    # # 根据实际高度图尺寸调整
                    # height_map_array = height_map_array.reshape((height_map_height, height_map_width))

                    if height_map_array is not None and save_data:
                        # 不保存为文件，直接使用 np.array
                        print("Received height map with shape:", height_map_array.shape)

            # 处理 JSON 数据
            json_start = bytes_data.find(b'Content-Type: application/json\r\n\r\n')
            if json_start != -1:
                json_data_start = json_start + len(b'Content-Type: application/json\r\n\r\n')
                end_json = bytes_data.find(b"\r\n--end\r\n")

                if end_json != -1:
                    json_buffer = bytes_data[json_data_start:end_json]
                    highstate = json.loads(json_buffer.decode('utf-8'))
                    # print("Received highstate:", highstate)
                    bytes_data = bytes_data[end_json + len(b"\r\n--end\r\n"):]

            if end_of_data != -1:
                break

        print(height_map_data["origin"])
        return {"height_map": height_map_data, "highstate": highstate}

    def move_to_xy(self, motion_planner):
        for motion in motion_planner:
            self.move(motion)
            time.sleep(motion.time)

    def avoid_obstacle_with_hybrid_astar(self, goal):
        while True:
            temp_num = 0
            # while True:
            height_map = self.fetch_height_map_data_with_highstate()
            # bev_obstacle = transform_heightmap2bevocc(height_map["height_map"]['data'])
            bev_obstacle = transform_heightmap2bevocc(height_map["height_map"]['data'],temp_num)
            temp_num = temp_num + 1
            go2_locate = self.fetch_xyyaw()
            motion_planner = hybrid_a_star(go2_locate,goal,bev_obstacle)

            self.move_to_xy(motion_planner)

        return "Succcess"

    def local_navigation(self, paths):
        for path_ in paths:
            state = self.avoid_obstacle_with_hybrid_astar(path_)


def transform_heightmap2bevocc(height_map, temp_num, threshold=0.1):
    # y_move, x_move = origin
    # y_move, x_move =  y_move / resolution,  x_move / resolution
    # print("y,x",y_move,x_move)
    # height_map_ = np.zeros([128,128])

    # # 使用切片进行移动
    # # if y_move == 0 and x_move != 0:  # 左右移动
    # if x_move > 0:  # 向右移动
    #     height_map_[:, :-int(x_move)] = height_map[:, int(x_move):]
    # elif x_move < 0:  # 向左移动
    #     height_map_[:, int(-x_move):] = height_map[:, :-int(-x_move)]
    # height_map = height_map_
    # height_map_ = np.zeros([128,128])
    # # elif y_move != 0 and x_move == 0:  # 上下移动
    # if y_move > 0:  # 向下移动
    #     height_map_[:-int(y_move), :] = height_map[int(y_move):, :]
    # elif y_move < 0:  # 向上移动
    #     height_map_[int(-y_move):, :] = height_map[:-int(-y_move), :]

    
    bev_obstacle_map = ((height_map > threshold) & (height_map < 9.9e+8)).astype(np.uint8)

    cv2.imwrite("height_map{}.png".format(temp_num), (height_map * 255 / height_map.max()).astype(np.uint8))  # 将高度图归一化到 0-255
    # 保存障碍物图
    cv2.imwrite("obstacle_map{}.png".format(temp_num), bev_obstacle_map * 255)  # 将障碍物图转换为 0-255 格式

    return bev_obstacle_map



if __name__ == '__main__':
    import math
    import time
    from src.utils import move_to_xy_with_yaw, quaternion_to_yaw, move_to_xy, get_delta_yaw
    from src.go2 import world_to_go2
    go2 = Go2()
    
    while True:
        rgbd_with_highstate = go2.fetch_rgbd_with_highstate()
        position = rgbd_with_highstate['highstate']['position'] # [x, y, z]
        quaternion = rgbd_with_highstate['highstate']["imu_quaternion"]
        yaw = quaternion_to_yaw(quaternion)
        print(yaw)

    # pts = np.array([position[0], position[1]])
    # next_pts = np.array([0,0])

    # print("current pts:", pts, "current yaw:", yaw)
    # print("next pts:", next_pts, "next yaw:", 0)

    # world_next_pts = world_to_self(pts, next_pts, yaw)
    # print("next pts:", next_pts, "----to-self---->", world_next_pts)
    # vx, vy, duration = move_to_xy(np.array([0,0]), np.array(world_next_pts))
    # print(f"vx: {vx}, vy: {vy}, duration: {duration}")

    # # import pdb; pdb.set_trace()
    # # go2.velocity_move(vx, vy, 0, duration)
    # # time.sleep(duration)
    # delta_yaw = get_delta_yaw(yaw, 0)
    # yaw_duration = go2.turn_vyaw(delta_yaw)
    # time.sleep(yaw_duration)
    # time.sleep(5)

    # rgbd_with_highstate = go2.fetch_rgbd_with_highstate()
    # position = rgbd_with_highstate['highstate']['position'] # [x, y, z]
    # yaw = quaternion_to_yaw(quaternion) # [x, y, z]
    # print("current pts:", [position[0], position[1]], "current yaw:", yaw)
