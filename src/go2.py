import numpy as np
import math

def pos_normal_to_go2(pts):
    T_robot_normal = np.array([
        [-1, 0,  0],  # X_normal -> Y_robot
        [0,  1,  0],  # Y_normal -> X_robot
        [0,  0,  1],  # Z_normal -> Z_robot
    ])
    return np.dot(T_robot_normal, pts)


def pos_go2_to_normal(pts):
    T_normal_robot = np.array([
        [-1, 0,  0],  # X_robot -> Y_normal
        [0,  1,  0],  # Y_robot -> -X_normal
        [0,  0,  1],  # Z_robot -> Z_normal
    ])
    return np.dot(T_normal_robot, pts)


def pose_go2_to_normal(pose):
    T_normal_robot = np.array([
        [0, -1,  0,  0],  # X_robot -> Y_normal
        [1,  0,  0,  0],  # Y_robot -> -X_normal
        [0,  0,  1,  0],  # Z_robot -> Z_normal
        [0,  0,  0,  1],  # 齐次坐标
    ])
    # 应用变换
    return np.dot(T_normal_robot, pose)


def pose_normal_to_tsdf(pose):
    return np.dot(
        pose, np.array([[ 1,  0,  0,  0], 
                        [ 0, -1,  0,  0], 
                        [ 0,  0, -1,  0], 
                        [ 0,  0,  0,  1]])
    )


def pose_normal_to_tsdf_real(pose):
    # This one makes sense, which is making x-forward, y-left, z-up to z-forward, x-right, y-down
    return pose @ np.array([[ 0,  0,  1,  0], 
                            [-1,  0,  0,  0], 
                            [ 0, -1,  0,  0], 
                            [ 0,  0,  0,  1]])


def world_to_go2(translation, world_point, theta):
    # translation: ndarray 平移量 世界坐标系下的ego位置
    # world_point: ndarray 世界坐标系下的目标点
    # theta: flaot 旋转角
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], 
                                [-np.sin(theta), np.cos(theta)]])

    ego_point = rotation_matrix.dot(world_point - translation)
    return ego_point

if __name__=="__main__":
    ego_point = world_to_go2(np.array([1,1]), np.array([3,3]), np.pi/4)
    print(ego_point)