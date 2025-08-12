# evaluate_para_eqa.py

"""
1. 在run_para_eqa.py运行完成后，根据results统计EQA信息
2. 统计questions_init正确率、questions_follow_up正确率、总正确率、直接回答率
3. 统计每一组问题的平均规范化步数、最大步数、最小步数
4. 针对数据集中的所有问题组，计算总体正确率、步数、直接回答率等指标
5. 输出统计结果到一个新的文件中
"""

import os
import json
import yaml
import glob
import numpy as np
import warnings
from collections import OrderedDict


def calculate_metrics(group_results, ground_truth):
    """计算单个问题组的评估指标"""
    
    # 将ground_truth问题映射到description以便快速查找
    gt_map = {}
    init_questions_gt = {q['question']: q['answer'] for q in ground_truth.get('questions_init', []) or []}
    follow_up_questions_gt = {q['question']: q['answer'] for q in ground_truth.get('questions_follow_up', []) or []}
    
    all_gt_questions = {**init_questions_gt, **follow_up_questions_gt}

    # 初始化统计变量
    init_correct = 0
    init_total = 0
    follow_up_correct = 0
    follow_up_total = 0
    direct_answers = 0
    normalized_steps = []

    for result in group_results:
        desc = result['description']
        
        # 检查这个问题是否存在于ground truth中
        if desc not in all_gt_questions:
            warnings.warn(f"问题 '{desc[:50]}...' 在结果文件中找到，但在基准文件中未找到。")
            continue
            
        is_correct = (result['answer'] == all_gt_questions[desc])
        
        # 判断是初始问题还是跟进问题
        if desc in init_questions_gt:
            init_total += 1
            if is_correct:
                init_correct += 1
        elif desc in follow_up_questions_gt:
            follow_up_total += 1
            if is_correct:
                follow_up_correct += 1

        # 处理步数和直接回答
        if result['max_steps'] == 0:
            direct_answers += 1
        else:
            norm_step = result['used_steps'] / result['max_steps']
            normalized_steps.append(norm_step)

    # 检查是否有问题在基准中但不在结果中
    for gt_q in all_gt_questions:
        if not any(res['description'] == gt_q for res in group_results):
            warnings.warn(f"基准问题 '{gt_q[:50]}...' 在结果文件中缺失，将被忽略。")

    # 计算最终指标
    total_evaluated = init_total + follow_up_total
    if total_evaluated == 0:
        return None

    metrics = {
        'questions_init_accuracy': init_correct / init_total if init_total > 0 else 0,
        'questions_follow_up_accuracy': follow_up_correct / follow_up_total if follow_up_total > 0 else 0,
        'total_accuracy': (init_correct + follow_up_correct) / total_evaluated,
        'direct_answer_rate': direct_answers / total_evaluated,
        'avg_normalized_steps': np.mean(normalized_steps) if normalized_steps else 0,
        'max_normalized_steps': np.max(normalized_steps) if normalized_steps else 0,
        'min_normalized_steps': np.min([s for s in normalized_steps if s > 0]) if any(s > 0 for s in normalized_steps) else 0,
        'evaluated_question_count': total_evaluated
    }
    
    return metrics


def main():
    benchmark_dir = 'data/benchmark'
    results_dir = 'results'
    output_file = os.path.join(results_dir, 'evaluation.json')

    benchmark_files = glob.glob(os.path.join(benchmark_dir, 'G*.yaml'))
    
    evaluation_results = {}

    for benchmark_file in benchmark_files:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            ground_truth_data = yaml.safe_load(f)
        
        group_id = ground_truth_data.get('group_id')
        if group_id is None:
            warnings.warn(f"基准文件 {benchmark_file} 缺少 'group_id'。")
            continue

        result_file = os.path.join(results_dir, f'answers_{group_id}.json')

        if not os.path.exists(result_file):
            warnings.warn(f"未找到组 '{group_id}' 对应的结果文件 {result_file}。")
            continue

        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                results_data = json.load(f)
            except json.JSONDecodeError:
                warnings.warn(f"无法解析结果文件 {result_file}。")
                continue
        
        print(f"正在评估组: {group_id}")
        group_metrics = calculate_metrics(results_data, ground_truth_data)
        
        if group_metrics:
            evaluation_results[group_id] = group_metrics

    # 计算总体指标
    if evaluation_results:
        total_accuracy_sum = 0
        total_norm_steps_sum = 0
        total_direct_answer_rate_sum = 0
        num_groups = 0

        for group_id, metrics in evaluation_results.items():
            total_accuracy_sum += metrics['total_accuracy']
            total_norm_steps_sum += metrics['avg_normalized_steps']
            total_direct_answer_rate_sum += metrics['direct_answer_rate']
            num_groups += 1
            
        overall_metrics = {
            'overall_accuracy': total_accuracy_sum / num_groups if num_groups > 0 else 0,
            'overall_avg_normalized_steps': total_norm_steps_sum / num_groups if num_groups > 0 else 0,
            'overall_direct_answer_rate': total_direct_answer_rate_sum / num_groups if num_groups > 0 else 0,
            'evaluated_group_count': num_groups
        }
        
        # 按组号升序排序，'overall'放在最前面
        sorted_results = OrderedDict()
        sorted_results['overall'] = overall_metrics
        for group_id in sorted([gid for gid in evaluation_results if gid != 'overall'], key=lambda x: int(x) if str(x).isdigit() else x):
            sorted_results[group_id] = evaluation_results[group_id]
        evaluation_results = sorted_results

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\n评估完成。结果已保存到 {output_file}")


if __name__ == '__main__':
    main()
