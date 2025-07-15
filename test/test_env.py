# @title Configure Sim Settings
import habitat_sim
import numpy as np
import random
import matplotlib.pyplot as plt

# 首先定义要探索的环境，包含了要探索房屋的网格和语义信息
test_scene = "./data/HM3D/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb"

# 设置观测空间，是否需要使用相关的传感器
rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}

def display_sample(rgb, depth, save_path="sample.png"):
    # 创建一个包含3列的子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

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

    # 显示图像
    plt.show()


# 定义传感器的一些属性
sim_settings = {
    "width": 256,  # Spatial resolution of the observations，分辨率
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0, # 设置默认agent为0
    "sensor_height": 1.5,  # Height of sensors in meters，高度
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation，设置伪随机数字再生种子
    "enable_physics": False,  # kinematics only，是否需要启动交互性，对于导航任务不需要，对于重排任务需要
}

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration() # 获取一个全局配置的结构体
    sim_cfg.gpu_device_id = 0 # 在0号gpu上进行配置
    sim_cfg.scene_id = settings["scene"] # 场景设置
    sim_cfg.enable_physics = settings["enable_physics"] # 物理功能是否启用，默认情况下为false

    # Note: all sensors must have the same resolution
    sensor_specs = []
	
	# 创建实例然后填充传感器配置参数
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor" # uuid必须唯一
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec) # 将color_sensor_spec加入到sensor_specs中

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec) # 将sensor_specs加入到sensor_specs中

    # Here you can specify the amount of displacement in a forward action and the turn angle
    # 有了传感器后必须将其加到agent上
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    # 定义agent的观测空间
    # 给agent配置所有的传感器
    agent_cfg.sensor_specifications = sensor_specs
    # 定义agent的动作空间
    # agent可以做到的动作，是一个字典包含前进、左转和右转
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25) # 向前进0.25m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0) # 向左转30度
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0) # 向右转30度
        ),
    }
	# 返回habita的全局配置，sim_cfg环境配置，以及agent的配置[agent_cfg]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# 运行刚刚定义的函数
cfg = make_cfg(sim_settings)
# Needed to handle out of order cell run in Colab
try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)

def print_scene_recur(scene, limit_output=10):
    # 打印几层楼、房间中的区域和和房间中的物体
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    # 打印bounding box的center和sizes
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    # 循环遍历这些楼层
    for level in scene.levels:
        # 获取id号、aabb的center和aabb的sizes
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        # 可视化语义传感器信息
        for region in level.regions:
            # 一个房间，每个房间都有类别名称
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            # 房间中的物体和物体的位置
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return


# Print semantic annotation information (id, category, bounding box details)
# about levels, regions and objects in a hierarchical fashion
scene = sim.semantic_scene # 获取语义场景
print_scene_recur(scene) # 查看场景周围存在的物品，并打印出场景中的内容

# the randomness is needed when choosing the actions
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space，设置agent在世界坐标系下的位置
agent.set_state(agent_state)

# Get agent state
# 打印状态确保agent实际能够正常工作
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

total_frames = 0
# 列出对于默认agent所有可能的动作
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

max_frames = 5

# 共有5个帧循环
while total_frames < max_frames:
    # 随机选择动作
    action = random.choice(action_names)
    print("action", action)
    # 执行动作
    observations = sim.step(action)
    # 列出所有的rgb、semantic和depth的观测结果
    rgb = observations["color_sensor"]
    depth = observations["depth_sensor"]
    print(sim.position)
    display_sample(rgb, depth)

    total_frames += 1
