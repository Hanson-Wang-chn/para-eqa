from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt

image_path = "results/diff-mt/Qwen2-VL-7B-Instruct/Qwen2-VL-7B-Instruct_gpu0/0/0.png"

detector = YOLO('/home/zml/model/YOLO/yolo11x.pt')  # initialize
results = detector.predict(image_path)

# Load image
image = Image.open(image_path).convert("RGB")
# image = Image.fromarray(rgb, mode="RGBA")

# Plot image
plt.imshow(image)

# Plot bounding boxes
result = results[0]
cls = result.names

for box in result.boxes:
    print(cls[box.cls.item()])
    box = box.xyxy[0].cpu()
    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))

plt.axis('off')
plt.savefig("1.jpg")
plt.show()