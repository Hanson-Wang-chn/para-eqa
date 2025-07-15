import os
import json
import numpy as np

# {
#     "meta": {
#         "question": "What is the relative volume of the SMX in the SMX?",
#         "scene": "FloorPlan1",
#         "floor": "1",
#         "max_steps": 4
#     },
#     "question_ind": 0,
#     "step_0": {
#         "step": 0,
#         "pts": [
#             1.551065,
#             0.13908827,
#             -3.0018978
#         ],
#         "angle": 0.0953831335597118,
#         "smx_vlm_rel": "No",
#         "smx_vlm_pred": "C",
#         "is_succeess": false
#     },
# }

def load_jsons(files_path):
    """
    读取 JSON 文件。

    :param file_path: 文件路径
    :return: JSON 数据
    """
    all_data = []
    for file_path in files_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                all_data.extend(json.load(file))
    return all_data

def evaluate(files_path):
    samples = load_jsons(files_path)
    results = []
    choices = ['a', 'b', 'c', 'd']
    for data in samples:
        result = {}

        meta_data = data['meta']
        step_data = data['step']
        max_step = meta_data['max_steps']
        answer = meta_data['answer']

        for step in step_data:
            step_num = step['step'] + 1
            res = step['smx_vlm_pred']
            if isinstance(res, str):
                res = step['smx_vlm_pred'].split('. ')[0]
                if step['smx_vlm_pred'].split('. ')[0].lower() == "y":
                    res = "yes"
                elif step['smx_vlm_pred'].split('. ')[0].lower() == "n":
                    res = "no"
            elif isinstance(res, list):
                index = np.array(np.argmax(res))
                res = choices[index]
            is_success = answer.lower() == res
            if is_success and "norm_early_success_step" not in result.keys():
                result["norm_early_success_step"] = step_num / max_step
            result["norm_success_step"] = step_num / max_step
            result["is_success"] = is_success
        results.append(result)

    results_num = len(results)
    norm_steps = 0
    norm_early_steps = 0
    early_count = 0
    success_count = 0
    for result in results:
        if result.get("is_success"):
            success_count += 1
            norm_steps += result.get("norm_success_step")
        if result.get("norm_early_success_step"):
            norm_early_steps += result.get("norm_early_success_step")
            early_count += 1

    print(f"总共有{results_num}个结果，其中成功的有{success_count}个。")
    print(f"成功率为{success_count/results_num:.2%}。")
    print(f"平均归一化成功步数为{norm_steps/success_count:.2}。")
    if early_count > 0:
        print(f"总共有{early_count}个结果提成功。")
        print(f"提早成功率为{early_count/results_num:.2%}")
        print(f"平均归一化提早成功步数为{norm_early_steps/early_count:.2}。")

if __name__ == '__main__':
    files_path = [
        'results/full-memory/HM-EQA/HM-EQA_gpu0/results.json',
        "results/full-memory/HM-EQA/HM-EQA_gpu3/results-480.json",
        "results/full-memory/HM-EQA/HM-EQA_gpu2/results-360.json"
    ]
    evaluate(files_path)
