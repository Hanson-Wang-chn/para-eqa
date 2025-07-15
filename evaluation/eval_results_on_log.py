import os

def find_lines_with_prefix(file_path, prefixes):
    """
    查找文本文件中以指定字符串前缀列表中任意一个开头的所有行。

    :param file_path: 文件路径
    :param prefixes: 指定的字符串前缀列表
    :return: 包含符合条件的行的列表
    """
    matching_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if any(line.startswith(prefix) for prefix in prefixes):
                matching_lines.append(line.strip())
            if line.startswith("Index: "):
                for _ in range(8):
                    matching_lines.append(next(file).strip())
            if line.startswith("== step:"):
                for _ in range(3):
                    matching_lines.append(next(file).strip())
    return matching_lines

def multi_file_evaluation(files_path: list, prefixes: list):
    """
    多文件评估，统计多个文件中以指定字符串前缀列表中任意一个开头的行的数量。

    :param files_path: 文件路径列表
    :param prefixes: 指定的字符串前缀列表
    :return: 包含符合条件的行的列表
    """
    matching_lines = []
    for file_path in files_path:
        matching_lines.extend(find_lines_with_prefix(file_path, prefixes))

    # print(matching_lines)
    results = []
    result_dict = {}
    success_count = 0
    for idx, line in enumerate(matching_lines, 1):
        log = line.split(' ')
        if line.startswith("Index:"):
            result_dict["index"] = int(log[1])
            result_dict["scene"] = log[3]
            result_dict["floor"] = log[5]
            result_dict["question"] = matching_lines[idx + 1]
            i = 0
            while True:
                i += 1
                if matching_lines[idx + i].startswith("Scene size:"):
                    result_dict["max_step"] = int(matching_lines[idx + i].split(' ')[-1])
                    break
        elif line.startswith("== step:"):
            step = int(log[-1]) + 1
            result_dict["last_step"] = step
            step_response = matching_lines[idx + 2].split(' ')[-1]
            if step_response == "True" and "norm_early_success_step" not in result_dict.keys():
                result_dict["norm_early_success_step"] = step / result_dict.get("max_step")
        elif line.startswith("Success"):
            result_dict["is_success"] = True if log[-1] == "True" else False
            result_dict["norm_success_step"] = result_dict.get("last_step") / result_dict.get("max_step")

            results.append(result_dict)
            result_dict = {}

    results_num = len(results)
    norm_steps = 0
    norm_early_steps = 0
    early_count = 0
    for result in results:
        if result.get("is_success"):
            success_count += 1
            norm_steps += result.get("norm_success_step")
        if result.get("norm_early_success_step"):
            norm_early_steps += result.get("norm_early_success_step")
            early_count += 1

    print(f"总共有{results_num}个结果，其中成功的有{success_count}个。")
    print(f"成功率为{success_count/results_num:.2%}。")
    print(f"归一化平均成功步数为{norm_steps/success_count:.2}。")
    if early_count > 0:
        print(f"总共有{early_count}个结果提早成功。")
        print(f"提早成功率为{early_count/results_num:.2%}")
        print(f"提早成功归一化步数为{norm_early_steps/early_count:.2}。")
    elif early_count == 0:
        print("没有提早成功的结果。")

    return matching_lines

if __name__ == '__main__':
    # 示例用法
    files_path = [
        'results/4.qwen2vl_rag_allsteps/qwen_rag_allsteps_gpu0/log_0.log',
        'results/4.qwen2vl_rag_allsteps/qwen_rag_allsteps_gpu1/log_1.log',
        'results/4.qwen2vl_rag_allsteps/qwen_rag_allsteps_gpu2/log_2.log',
        'results/4.qwen2vl_rag_allsteps/qwen_rag_allsteps_gpu3/log_3.log',
        'results/4.qwen2vl_rag_allsteps/qwen_rag_allsteps_gpu4/log_4.log',
        'results/4.qwen2vl_rag_allsteps/qwen_rag_allsteps_gpu5/log_5.log',
    ]
    prefixes = ['Index: ', 
                '== step:', 
                'Success (max):'
                ]
    matching_lines = multi_file_evaluation(files_path, prefixes)
