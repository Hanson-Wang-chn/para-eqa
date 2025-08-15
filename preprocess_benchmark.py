# preprocess_benchmark.py

import os
import sys
import glob
import yaml
import random
import argparse
from typing import Dict, Any, List


def validate_yaml_file(data: Dict[str, Any]) -> bool:
    """验证YAML文件格式是否符合要求"""
    # 检查必要字段
    required_fields = ['group_id', 'scene', 'floor', 'init_x', 'init_y', 'init_z', 'init_angle']
    for field in required_fields:
        if field not in data or data[field] is None:
            print(f"错误: 缺失或为空的必要字段 '{field}'")
            return False
    
    # 检查questions_init和questions_follow_up字段
    if 'questions_init' not in data and 'questions_follow_up' not in data:
        print("错误: 'questions_init' 和 'questions_follow_up' 同时缺失")
        return False
    
    # 如果存在questions_init，检查其格式
    if 'questions_init' in data and data['questions_init']:
        for q in data['questions_init']:
            if 'question' not in q or 'answer' not in q or 'urgency' not in q:
                print("错误: questions_init 中包含没有 'question' 或 'answer' 或 'urgency' 字段的项")
                return False
            if q['urgency'] is None or q['urgency'] == '':
                print("错误: questions_init 中的 'urgency' 字段不能为空")
                return False
            if not isinstance(q['urgency'], (int, float)):
                print("错误: questions_init 中的 'urgency' 字段必须是数字")
                return False
            if q['urgency'] < 0 or q['urgency'] > 1:
                print("错误: questions_init 中的 'urgency' 字段必须在0到1之间")
                return False
            if not isinstance(q['question'], str) or not isinstance(q['answer'], str):
                print("错误: questions_init 中的 'question' 和 'answer' 字段必须是字符串")
                return False
            if not q['question'].strip() or not q['answer'].strip():
                print("错误: questions_init 中的 'question' 和 'answer' 字段不能为空")
                return False
    
    # 如果存在questions_follow_up，检查其格式
    if 'questions_follow_up' in data and data['questions_follow_up']:
        for q in data['questions_follow_up']:
            if 'question' not in q or 'answer' not in q or 'urgency' not in q:
                print("错误: questions_follow_up 中包含没有 'question' 或 'answer' 或 'urgency' 字段的项")
                return False
            if q['urgency'] is None or q['urgency'] == '':
                print("错误: questions_follow_up 中的 'urgency' 字段不能为空")
                return False
            if not isinstance(q['urgency'], (int, float)):
                print("错误: questions_follow_up 中的 'urgency' 字段必须是数字")
                return False
            if q['urgency'] < 0 or q['urgency'] > 1:
                print("错误: questions_follow_up 中的 'urgency' 字段必须在0到1之间")
                return False
            if not isinstance(q['question'], str) or not isinstance(q['answer'], str):
                print("错误: questions_follow_up 中的 'question' 和 'answer' 字段必须是字符串")
                return False
            if not q['question'].strip() or not q['answer'].strip():
                print("错误: questions_follow_up 中的 'question' 和 'answer' 字段不能为空")
                return False
    
    return True

def shuffle_questions(data: Dict[str, Any]) -> Dict[str, Any]:
    """重新随机排序问题"""
    # 获取原始问题数量
    init_count = len(data.get('questions_init', []))
    follow_up_count = len(data.get('questions_follow_up', []))
    
    # 收集所有问题
    all_questions = []
    if 'questions_init' in data and data['questions_init']:
        all_questions.extend(data['questions_init'])
    if 'questions_follow_up' in data and data['questions_follow_up']:
        all_questions.extend(data['questions_follow_up'])
    
    # 如果没有问题，直接返回
    if not all_questions:
        return data
    
    # 随机打乱问题
    random.shuffle(all_questions)
    
    # 重新分配问题
    new_data = data.copy()
    if init_count > 0:
        new_data['questions_init'] = all_questions[:init_count]
    if follow_up_count > 0:
        new_data['questions_follow_up'] = all_questions[init_count:init_count+follow_up_count]
    
    return new_data

def main():
    parser = argparse.ArgumentParser(description='处理基准测试YAML文件。')
    parser.add_argument('--reorder', default=True, help='随机重排问题')
    parser.add_argument('--dir', default='data/benchmark', help='包含YAML文件的目录')
    args = parser.parse_args()
    
    # 是否随机重排问题
    reorder_randomly = args.reorder
    benchmark_dir = args.dir
    
    # 获取目录下所有YAML文件并排序
    yaml_files = sorted(glob.glob(f'{benchmark_dir}/*.yaml'))
    
    if not yaml_files:
        print(f"错误: 在 {benchmark_dir} 目录中未找到YAML文件")
        sys.exit(1)
    
    print(f"在基准测试目录中找到 {len(yaml_files)} 个YAML文件")
    
    # TODO: 选择要处理的文件
    yaml_files = yaml_files[40:]
    
    if not yaml_files:
        print(f"错误: 没有需要处理的文件")
        sys.exit(1)
    
    print(f"开始处理 {len(yaml_files)} 个YAML文件")
    
    # 处理每个文件
    for yaml_file in yaml_files:
        print(f"处理 {yaml_file}...")
        
        try:
            # 读取YAML文件
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 验证文件格式
            if not validate_yaml_file(data):
                print(f"跳过无效文件: {yaml_file}")
                continue
            
            # 如果需要，随机重排问题
            if reorder_randomly:
                data = shuffle_questions(data)
                print(f"已为 {yaml_file} 重排问题")
            
            # 将处理后的内容写回文件
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            print(f"成功处理 {yaml_file}")
            
        except Exception as e:
            print(f"处理 {yaml_file} 时出错: {e}")
    
    print("处理完成")

if __name__ == "__main__":
    main()