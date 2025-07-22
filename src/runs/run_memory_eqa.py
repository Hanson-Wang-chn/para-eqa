"""
Run EQA in Habitat-Sim with VLM exploration.

"""
import os
import numpy as np
import logging
import csv
import json
import argparse

from tqdm import tqdm
from src.modeling.memory_eqa import MemoryEQA

os.environ["QT_QPA_PLATFORM"] = "offscreen"

np.set_printoptions(precision=3)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

def run_on_gpu(gpu_id, gpu_index, gpu_count, cfg_file):
    from omegaconf import OmegaConf
    """在指定 GPU 上运行 main(cfg)，并传递 GPU 信息"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 设置可见的 GPU
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, f"{cfg.exp_name}/{cfg.exp_name}_gpu{gpu_id}")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, f"log_{gpu_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # 将 GPU 信息传递给 main 函数
    logging.info(f"***** Running {cfg.exp_name} on GPU {gpu_id}/{gpu_count} *****")

    # print(cfg)
    # {'seed': 42, 'exp_name': 'HM-EQA', 'output_parent_dir': 'results/full-memory', 'question_data_path': './data/MT-HM3D/MT-HM3D-filtered-new.csv', 'init_pose_data_path': './data/scene_init_poses_all.csv', 'scene_data_path': ['./data/HM3D'], 'save_obs': True, 'save_freq': 20, 'vlm': {'device': 'cuda', 'model_name_or_path': 'gpt-4.1'}, 'rag': {'use_rag': True, 'text': 'openai/clip-vit-large-patch14', 'visual': 'openai/clip-vit-large-patch14', 'dim': 1536, 'max_retrieval_num': 5}, 'prompt': {'caption': 'f"Describe this image."', 'relevent': "\nConsider the question: '{}'. How confident are you in answering this question from your current perspective?\nA. Very low\nB. Low\nC. Medium\nD. High\nE. Very high\nAnswer with the option's letter from the given choices directly.", 'question': "{}\nAnswer with the option's letter from the given choices directly.", 'local_sem': "\nConsider the question: '{}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Provide reasons and answer with a single letter.", 'global_sem': "\nConsider the question: '{}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."}, 'detector': '/home/whs/eqa/para-eqa/yolo11x.pt', 'camera_height': 1.5, 'camera_tilt_deg': -30, 'img_width': 640, 'img_height': 480, 'hfov': 120, 'tsdf_grid_size': 0.1, 'margin_w_ratio': 0.25, 'margin_h_ratio': 0.6, 'init_clearance': 0.5, 'max_step_room_size_ratio': 3, 'black_pixel_ratio': 0.7, 'min_random_init_steps': 2, 'use_active': True, 'use_lsv': True, 'use_gsv': True, 'gsv_T': 0.5, 'gsv_F': 3, 'planner': {'dist_T': 10, 'unexplored_T': 0.2, 'unoccupied_T': 2.0, 'val_T': 0.5, 'val_dir_T': 0.5, 'max_val_check_frontier': 3, 'smooth_sigma': 5, 'eps': 1, 'min_dist_from_cur': 0.5, 'max_dist_from_cur': 3, 'frontier_spacing': 1.5, 'frontier_min_neighbors': 3, 'frontier_max_neighbors': 4, 'max_unexplored_check_frontier': 3, 'max_unoccupied_check_frontier': 1}, 'visual_prompt': {'cluster_threshold': 1.0, 'num_prompt_points': 3, 'num_max_unoccupied': 300, 'min_points_for_clustering': 3, 'point_min_dist': 2, 'point_max_dist': 10, 'cam_offset': 0.6, 'min_num_prompt_points': 2, 'circle_radius': 18}, 'output_dir': 'results/full-memory/HM-EQA/HM-EQA_gpu0'}
    
    memory_eqa = MemoryEQA(cfg, gpu_id)

    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    
    
    # questions_data = [
    #     {
    #         'scene': '00797-99ML7CGPqsQ',
    #         'floor': '0',
    #         'question': 'Is the door color darker than the ceiling color?',
    #         'choices': "['Yes', 'No', 'They are the same color', 'The ceiling is darker']",
    #         'question_formatted': 'Is the door color darker than the ceiling color? A) Yes B) No C) They are the same color D) The ceiling is darker. Answer:',
    #         'answer': 'A',
    #         'label': 'Comparison',
    #         'source_image': '00797-99ML7CGPqsQ_0.png'
    #     },
    #     ...
    # ]
    

    results_all = []
    part_data = len(questions_data) / gpu_count
    start_idx = int(part_data * gpu_index)
    end_idx = int(part_data * (gpu_index + 1))
    for question_ind in tqdm(range(start_idx, end_idx)):
        data = questions_data[question_ind]
        
        result = memory_eqa.run(data, question_ind)
        
        results_all.append(result)
        if question_ind % cfg.save_freq == 0:
            with open(os.path.join(cfg.output_dir, f"results-{question_ind}.json"), "w") as f:
                json.dump(results_all, f, indent=4)

    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(results_all, f, indent=4)


if __name__ == "__main__":
    from multiprocessing import Process, set_start_method

    # 设置多进程启动方式为 spawn
    set_start_method("spawn", force=True)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg_file", help="cfg file path", default="cfg/vlm_exp_ov.yaml", type=str)
    parser.add_argument("-gpus", "--gpu_ids", help="Comma-separated GPU IDs to use (e.g., '0,1,2')", type=str, default="0")
    args = parser.parse_args()

    # Get list of GPUs
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    gpu_count = len(gpu_ids)  # 计算 GPU 数量

    # Launch processes for each GPU
    processes = []
    for gpu_id in gpu_ids:
        gpu_index = gpu_ids.index(gpu_id)
        p = Process(target=run_on_gpu, args=(gpu_id, gpu_index, gpu_count, args.cfg_file))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
