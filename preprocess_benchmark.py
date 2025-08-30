# preprocess_benchmark.py

import os
import sys
import glob
import yaml
import random
import argparse
from typing import Dict, Any, List


def validate_yaml_file(data: Dict[str, Any]) -> bool:
    """Validate YAML file format to ensure it meets requirements"""
    # Check required fields
    required_fields = ['group_id', 'scene', 'floor', 'init_x', 'init_y', 'init_z', 'init_angle']
    for field in required_fields:
        if field not in data or data[field] is None:
            print(f"Error: Missing or empty required field '{field}'")
            return False
    
    # Check questions_init and questions_follow_up fields
    if 'questions_init' not in data and 'questions_follow_up' not in data:
        print("Error: Both 'questions_init' and 'questions_follow_up' are missing")
        return False
    
    # If questions_init exists, check its format
    if 'questions_init' in data and data['questions_init']:
        for q in data['questions_init']:
            if 'question' not in q or 'answer' not in q or 'urgency' not in q:
                print("Error: questions_init contains items without 'question' or 'answer' or 'urgency' fields")
                return False
            if q['urgency'] is None or q['urgency'] == '':
                print("Error: 'urgency' field in questions_init cannot be empty")
                return False
            if not isinstance(q['urgency'], (int, float)):
                print("Error: 'urgency' field in questions_init must be a number")
                return False
            if q['urgency'] < 0 or q['urgency'] > 1:
                print("Error: 'urgency' field in questions_init must be between 0 and 1")
                return False
            if not isinstance(q['question'], str) or not isinstance(q['answer'], str):
                print("Error: 'question' and 'answer' fields in questions_init must be strings")
                return False
            if not q['question'].strip() or not q['answer'].strip():
                print("Error: 'question' and 'answer' fields in questions_init cannot be empty")
                return False
    
    # If questions_follow_up exists, check its format
    if 'questions_follow_up' in data and data['questions_follow_up']:
        for q in data['questions_follow_up']:
            if 'question' not in q or 'answer' not in q or 'urgency' not in q:
                print("Error: questions_follow_up contains items without 'question' or 'answer' or 'urgency' fields")
                return False
            if q['urgency'] is None or q['urgency'] == '':
                print("Error: 'urgency' field in questions_follow_up cannot be empty")
                return False
            if not isinstance(q['urgency'], (int, float)):
                print("Error: 'urgency' field in questions_follow_up must be a number")
                return False
            if q['urgency'] < 0 or q['urgency'] > 1:
                print("Error: 'urgency' field in questions_follow_up must be between 0 and 1")
                return False
            if not isinstance(q['question'], str) or not isinstance(q['answer'], str):
                print("Error: 'question' and 'answer' fields in questions_follow_up must be strings")
                return False
            if not q['question'].strip() or not q['answer'].strip():
                print("Error: 'question' and 'answer' fields in questions_follow_up cannot be empty")
                return False
    
    return True

def shuffle_questions(data: Dict[str, Any]) -> Dict[str, Any]:
    """Randomly reorder questions"""
    # Get original question counts
    init_count = len(data.get('questions_init', []))
    follow_up_count = len(data.get('questions_follow_up', []))
    
    # Collect all questions
    all_questions = []
    if 'questions_init' in data and data['questions_init']:
        all_questions.extend(data['questions_init'])
    if 'questions_follow_up' in data and data['questions_follow_up']:
        all_questions.extend(data['questions_follow_up'])
    
    # If no questions, return directly
    if not all_questions:
        return data
    
    # Randomly shuffle questions
    random.shuffle(all_questions)
    
    # Redistribute questions
    new_data = data.copy()
    if init_count > 0:
        new_data['questions_init'] = all_questions[:init_count]
    if follow_up_count > 0:
        new_data['questions_follow_up'] = all_questions[init_count:init_count+follow_up_count]
    
    return new_data

def main():
    parser = argparse.ArgumentParser(description='Process benchmark YAML files.')
    parser.add_argument('--reorder', default=True, help='Randomly reorder questions')
    parser.add_argument('--dir', default='data/benchmark', help='Directory containing YAML files')
    args = parser.parse_args()
    
    # Whether to randomly reorder questions
    reorder_randomly = args.reorder
    benchmark_dir = args.dir
    
    # Get all YAML files in directory and sort
    yaml_files = sorted(glob.glob(f'{benchmark_dir}/*.yaml'))
    
    if not yaml_files:
        print(f"Error: No YAML files found in {benchmark_dir} directory")
        sys.exit(1)
    
    print(f"Found {len(yaml_files)} YAML files in benchmark directory")
    
    # Select files to process
    yaml_files = yaml_files[40:]
    
    if not yaml_files:
        print(f"Error: No files to process")
        sys.exit(1)
    
    print(f"Starting to process {len(yaml_files)} YAML files")
    
    # Process each file
    for yaml_file in yaml_files:
        print(f"Processing {yaml_file}...")
        
        try:
            # Read YAML file
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Validate file format
            if not validate_yaml_file(data):
                print(f"Skipping invalid file: {yaml_file}")
                continue
            
            # If needed, randomly reorder questions
            if reorder_randomly:
                data = shuffle_questions(data)
                print(f"Reordered questions for {yaml_file}")
            
            # Write processed content back to file
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            print(f"Successfully processed {yaml_file}")
            
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
    
    print("Processing completed")

if __name__ == "__main__":
    main()
