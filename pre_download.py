import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# import clip

def main():
    # 将模型缓存到 project_root/model_cache 下
    os.environ["HF_HOME"] = os.path.abspath("model_cache/hf")
    os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("model_cache/transformers")
    os.environ["TORCH_HOME"] = os.path.abspath("model_cache/torch")

    # 一次性下载所有 Qwen2-VL 模型
    vlm_model_ids = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
    ]
    for model_id in vlm_model_ids:
        print(f"Downloading VLM model {model_id} ...")
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
            resume_download=True,
            local_files_only=False,
            device_map="cpu",
        )
        AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=False,
        )
    print("✅ All VLM models downloaded.")

    # 一次性下载所有 CLIP 模型
    # clip_models = clip.available_models()
    # for name in clip_models:
    #     print(f"Downloading CLIP model {name} ...")
    #     clip.load(name, device="cpu", jit=False)
    # print("✅ All CLIP models downloaded.")

if __name__ == "__main__":
    main()
    