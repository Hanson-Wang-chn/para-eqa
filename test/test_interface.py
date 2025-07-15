import os
import requests
import json
import cv2

response = requests.get('http://10.24.6.240:5000/video_feed')

def fetch_video_with_highstate(self, save_data=False):
    highstate = None
    frame = None
    frame_number = 0
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
                frame_filename = f"images/frames/{frame_number:04d}.jpg"
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {frame_filename}")
                frame_number += 1

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
                    json_filename = f"images/json/{frame_number:04d}.json"
                    with open(json_filename, 'w') as f:
                        json.dump(highstate, f, indent=4)

        if end_of_data != -1:
            break

    return {"frame": frame, "highstate": highstate}


# print('Status Code:', response.status_code)
print('Response Text:', response)