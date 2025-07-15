import cv2
import os

def create_video(image_folder, video_name, fps):
    # 获取图片路径列表
    idx = 0
    images = []
    while True:
        idx += 1
        jdx = -1

        # while True:
        #     jdx += 1
        #     img_name = f"{idx}-{jdx}.png"
        #     img_path = os.path.join(image_folder, img_name)
        #     if os.path.exists(img_path):
        #         images.append(img_name)
        #     else:
        #         break

        img_name = f"{idx}_map.png"
        img_path = os.path.join(image_folder, img_name)
        if os.path.exists(img_path):
            images.append(img_name)
        else:
            break
    # images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    print(images)
    # images.sort()  # 如果需要按文件名排序，可以这样排序
    
    # 读取第一张图片的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    
    # 定义视频编码器，创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用mp4编码格式
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    
    # 将每张图片写入视频
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"无法读取图片: {img_path}")
        else:
            cv2.waitKey(0)
        video.write(frame)
    
    # 释放视频对象
    video.release()
    print(f"Video saved as {video_name}")

# 使用示例
image_folder = '/home/smbu/zml/algorithm/explore-eqa/results/vlm_exp_test/0'  # 替换为图片文件夹路径
video_name = '/home/smbu/zml/algorithm/explore-eqa/results/videos/00_map.mp4'  # 输出视频文件名
fps = 2  # 指定帧率
create_video(image_folder, video_name, fps)
