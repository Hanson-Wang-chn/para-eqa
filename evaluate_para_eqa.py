# evaluate_para_eqa.py

"""
1. After run_para_eqa.py is completed, collect EQA information based on results
2. Calculate accuracy of questions_init, accuracy of questions_follow_up, total accuracy, direct answer rate, and accuracy of direct answer questions
3. Calculate average normalized steps, maximum steps, and minimum steps for each question group
4. Calculate nuwl_time and nuwl_step for each question group
   - nuwl_time: Sum of the product of each question's urgency value and its waiting time (difference between "finish" and "request"), divided by the total number of questions in the group
   - nuwl_step: Sum of the product of each question's urgency value and its waiting step count, divided by the total number of questions in the group
5. Calculate overall accuracy, steps, direct answer rate, direct answer accuracy and other metrics for all question groups in the dataset
6. Output statistical results to a new file
"""

import os
import json
import yaml
import glob
import numpy as np
import warnings
from collections import OrderedDict


def calculate_metrics(group_results, ground_truth, enable_follow_up=True):
    """Calculate evaluation metrics for a single question group"""
    
    # Map ground_truth questions to description for quick lookup
    gt_map = {}
    init_questions_gt = {q['question']: {'answer': q['answer'], 'urgency': q['urgency']} 
                         for q in ground_truth.get('questions_init', []) or []}
    follow_up_questions_gt = {q['question']: {'answer': q['answer'], 'urgency': q['urgency']} 
                             for q in ground_truth.get('questions_follow_up', []) or []}
    
    all_gt_questions = {**init_questions_gt, **follow_up_questions_gt}

    # Initialize statistical variables
    init_correct = 0
    init_total = 0
    follow_up_correct = 0
    follow_up_total = 0
    direct_answers = 0
    direct_answers_correct = 0
    normalized_steps = []
    
    # For calculating nuwl_time and nuwl_step
    urgency_weighted_time_sum = 0.0
    urgency_weighted_step_sum = 0.0
    
    # Store information of all questions for calculating nuwl_step
    questions_info = []

    # First pass: collect basic question information
    for result in group_results:
        desc = result['description']
        
        # Check if this question exists in ground truth
        if desc not in all_gt_questions:
            warnings.warn(f"Question '{desc[:50]}...' found in result file but not found in benchmark file.")
            continue
            
        gt_answer = all_gt_questions[desc]['answer']
        urgency = all_gt_questions[desc]['urgency']
        is_correct = (result['answer'] == gt_answer)
        
        # Whether it's a direct answer question
        is_direct_answer = result['max_steps'] == -1 and result['used_steps'] == 0
        
        # Calculate normalized_steps
        norm_step = 0 if is_direct_answer else result['used_steps'] / result['max_steps']
        
        # Get time information
        request_time = result.get('time', {}).get('request', 0)
        start_time = result.get('time', {}).get('start', 0)
        finish_time = result.get('time', {}).get('finish', 0)
        
        # Add to question information list
        questions_info.append({
            'description': desc,
            'urgency': urgency,
            'is_correct': is_correct,
            'is_direct_answer': is_direct_answer,
            'is_init': desc in init_questions_gt,
            'normalized_steps': norm_step,
            'request_time': request_time,
            'start_time': start_time,
            'finish_time': finish_time
        })
        
        # Determine if it's an initial question or follow-up question
        if desc in init_questions_gt:
            init_total += 1
            if is_correct:
                init_correct += 1
        elif desc in follow_up_questions_gt:
            follow_up_total += 1
            if is_correct:
                follow_up_correct += 1

        # Handle steps and direct answers
        if is_direct_answer:
            direct_answers += 1
            if is_correct:
                direct_answers_correct += 1
        else:
            normalized_steps.append(norm_step)
        
        # Calculate nuwl_time (based on time)
        wait_time = finish_time - request_time
        urgency_weighted_time_sum += urgency * wait_time

    # Second pass: calculate nuwl_step
    if enable_follow_up:
        # Original logic, based on time dependencies
        for q in questions_info:
            # Find all other questions whose start time is earlier than this question's request time
            waiting_steps = q['normalized_steps']  # Its own steps
            
            for other_q in questions_info:
                # If another question's start time is earlier than current question's request time, its steps should also be counted in waiting
                if other_q != q and other_q['start_time'] < q['request_time']:
                    waiting_steps += other_q['normalized_steps']
            
            # Calculate urgency weighted steps for this question
            urgency_weighted_step_sum += q['urgency'] * waiting_steps
    
    else:
        # New logic, all questions are considered to enter at the same time
        # Create a dictionary mapping description to questions in questions_info
        desc_to_question = {q['description']: q for q in questions_info}
        
        # Process questions in the order of group_results
        cumulative_steps = 0
        for result in group_results:
            desc = result['description']
            if desc not in desc_to_question:
                continue
            
            q = desc_to_question[desc]
            
            # Waiting steps are cumulative steps plus its own steps
            waiting_steps = cumulative_steps + q['normalized_steps']
            urgency_weighted_step_sum += q['urgency'] * waiting_steps
            
            # Update cumulative steps
            cumulative_steps += q['normalized_steps']

    # Check if there are questions in benchmark but not in results
    for gt_q in all_gt_questions:
        if not any(q['description'] == gt_q for q in questions_info):
            warnings.warn(f"Benchmark question '{gt_q[:50]}...' is missing in result file and will be ignored.")

    # Calculate final metrics
    total_evaluated = init_total + follow_up_total
    if total_evaluated == 0:
        return None

    # Calculate nuwl_time and nuwl_step
    # nuwl_time = urgency_weighted_time_sum / total_evaluated if total_evaluated > 0 else 0
    nuwl_step = urgency_weighted_step_sum / total_evaluated if total_evaluated > 0 else 0

    metrics = {
        'questions_init_accuracy': init_correct / init_total if init_total > 0 else 0,
        'questions_follow_up_accuracy': follow_up_correct / follow_up_total if follow_up_total > 0 else 0,
        'total_accuracy': (init_correct + follow_up_correct) / total_evaluated,
        'direct_answer_rate': direct_answers / total_evaluated,
        'direct_answer_accuracy': direct_answers_correct / direct_answers if direct_answers > 0 else None,
        'avg_normalized_steps': np.mean(normalized_steps) if normalized_steps else 0,
        'max_normalized_steps': np.max(normalized_steps) if normalized_steps else 0,
        'min_normalized_steps': np.min([s for s in normalized_steps if s > 0]) if any(s > 0 for s in normalized_steps) else 0,
        # 'nuwl_time': nuwl_time,
        'nuwl_step': nuwl_step,
        'evaluated_question_count': total_evaluated,
        'direct_answer_count': direct_answers,
        'direct_answer_correct_count': direct_answers_correct
    }
    
    return metrics


def main(results_dir, output_file, enable_follow_up):
    benchmark_dir = 'data/benchmark'
    benchmark_files = glob.glob(os.path.join(benchmark_dir, 'G*.yaml'))
    
    # Sort files by group number in lexicographical order
    def extract_group_number(filename):
        basename = os.path.basename(filename)
        if basename.startswith('G') and basename.endswith('.yaml'):
            group_part = basename[1:-5]  # Remove 'G' and '.yaml'
            return group_part  # Return string, sort by lexicographical order
        return basename  # Other cases sort by complete filename lexicographically
    
    benchmark_files.sort(key=extract_group_number)
    
    evaluation_results = {}
    
    # Variables for calculating overall metrics
    total_questions = 0
    total_direct_answers = 0
    total_direct_answers_correct = 0
    total_urgency_weighted_time_sum = 0.0
    total_urgency_weighted_step_sum = 0.0

    for benchmark_file in benchmark_files:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            ground_truth_data = yaml.safe_load(f)
        
        group_id = ground_truth_data.get('group_id')
        if group_id is None:
            warnings.warn(f"Benchmark file {benchmark_file} is missing 'group_id'.")
            continue

        result_file = os.path.join(results_dir, f'answers_{group_id}.json')

        if not os.path.exists(result_file):
            warnings.warn(f"Result file {result_file} for group '{group_id}' not found.")
            continue

        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                results_data = json.load(f)
            except json.JSONDecodeError:
                warnings.warn(f"Cannot parse result file {result_file}.")
                continue
        
        print(f"Evaluating group: {group_id}")
        group_metrics = calculate_metrics(results_data, ground_truth_data, enable_follow_up)
        
        if group_metrics:
            evaluation_results[group_id] = group_metrics
            
            # Collect data needed for overall metrics
            total_questions += group_metrics['evaluated_question_count']
            total_direct_answers += group_metrics['direct_answer_count']
            total_direct_answers_correct += group_metrics['direct_answer_correct_count']
            # total_urgency_weighted_time_sum += group_metrics['nuwl_time'] * group_metrics['evaluated_question_count']
            total_urgency_weighted_step_sum += group_metrics['nuwl_step'] * group_metrics['evaluated_question_count']

    # Calculate overall metrics
    if evaluation_results:
        total_accuracy_sum = 0
        total_norm_steps_sum = 0
        total_direct_answer_rate_sum = 0
        total_nuwl_time_sum = 0
        total_nuwl_step_sum = 0
        num_groups = 0

        for group_id, metrics in evaluation_results.items():
            total_accuracy_sum += metrics['total_accuracy']
            total_norm_steps_sum += metrics['avg_normalized_steps']
            total_direct_answer_rate_sum += metrics['direct_answer_rate']
            # total_nuwl_time_sum += metrics['nuwl_time']
            total_nuwl_step_sum += metrics['nuwl_step']
            num_groups += 1
            
        overall_metrics = {
            'overall_accuracy': total_accuracy_sum / num_groups if num_groups > 0 else 0,
            'overall_avg_normalized_steps': total_norm_steps_sum / num_groups if num_groups > 0 else 0,
            'overall_direct_answer_rate': total_direct_answer_rate_sum / num_groups if num_groups > 0 else 0,
            'overall_direct_answer_accuracy': total_direct_answers_correct / total_direct_answers if total_direct_answers > 0 else 0,
            # 'overall_nuwl_time': total_urgency_weighted_time_sum / total_questions if total_questions > 0 else 0,
            'overall_nuwl_step': total_urgency_weighted_step_sum / total_questions if total_questions > 0 else 0,
            'evaluated_group_count': num_groups,
            'total_questions': total_questions,
            'total_direct_answers': total_direct_answers
        }
        
        # Sort by group number in ascending order, put 'overall' at the front
        sorted_results = OrderedDict()
        sorted_results['overall'] = overall_metrics
        for group_id in sorted([gid for gid in evaluation_results if gid != 'overall'], key=lambda x: int(x) if str(x).isdigit() else x):
            sorted_results[group_id] = evaluation_results[group_id]
        evaluation_results = sorted_results

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation completed. Results saved to {output_file}")


if __name__ == '__main__':
    # Parallel EQA
    results_dir_major = 'results/answers-major'
    output_file_major = 'results/evaluation-major.json'
    enable_follow_up_major = True
    main(results_dir_major, output_file_major, enable_follow_up_major)
    
    # Explore EQA
    results_dir_explore_eqa = 'results/answers-explore-eqa'
    output_file_explore_eqa = 'results/evaluation-explore-eqa.json'
    enable_follow_up_explore_eqa = False
    main(results_dir_explore_eqa, output_file_explore_eqa, enable_follow_up_explore_eqa)
    
    # Memory EQA
    results_dir_memory_eqa = 'results/answers-memory-eqa'
    output_file_memory_eqa = 'results/evaluation-memory-eqa.json'
    enable_follow_up_memory_eqa = False
    main(results_dir_memory_eqa, output_file_memory_eqa, enable_follow_up_memory_eqa)
    
    # Parallel EQA without priority
    results_dir_no_priority = 'results/answers-no-priority'
    output_file_no_priority = 'results/evaluation-no-priority.json'
    enable_follow_up_no_priority = True
    main(results_dir_no_priority, output_file_no_priority, enable_follow_up_no_priority)
    
    # Parallel EQA without urgency
    results_dir_no_urgency = 'results/answers-no-urgency'
    output_file_no_urgency = 'results/evaluation-no-urgency.json'
    enable_follow_up_no_urgency = True
    main(results_dir_no_urgency, output_file_no_urgency, enable_follow_up_no_urgency)
    
    # Parallel EQA without scope
    results_dir_no_scope = 'results/answers-no-scope'
    output_file_no_scope = 'results/evaluation-no-scope.json'
    enable_follow_up_no_scope = True
    main(results_dir_no_scope, output_file_no_scope, enable_follow_up_no_scope)
    
    # Parallel EQA without reward
    results_dir_no_reward = 'results/answers-no-reward'
    output_file_no_reward = 'results/evaluation-no-reward.json'
    enable_follow_up_no_reward = True
    main(results_dir_no_reward, output_file_no_reward, enable_follow_up_no_reward)
    
    # Parallel EQA without dependency
    results_dir_no_dependency = 'results/answers-no-dependency'
    output_file_no_dependency = 'results/evaluation-no-dependency.json'
    enable_follow_up_no_dependency = True
    main(results_dir_no_dependency, output_file_no_dependency, enable_follow_up_no_dependency)
