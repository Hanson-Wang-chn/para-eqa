# ParaEQA: Parallel and Asynchronous Embodied Question Answering

## Installation
Set up the conda environment (Linux, Python 3.9):
```
conda env create -f environment.yml
conda activate para-eqa
pip install -e .
```

If an error occurs when installing `flash-attention2`, you can try the [release version](https://github.com/Dao-AILab/flash-attention/releases) that suits your environment. For example:

```
pip install /path/to/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl --no-build-isolation
```

Install transformers for qwenvl
```
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
```

Install [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
```
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install -vvv --no-build-isolation -e .
```

## Project Structure

Overall structure:

```
.
├── assets
├── cfg
├── CLIP
├── data
├── environment.yml
├── evaluation
├── memory_eqa.egg-info
├── pre_download.py
├── README.md
├── results
├── scripts
├── server_wrapper
├── setup.py
├── src
├── test
├── tools
└── yolo11x.pt
```

`data` structure:

```
.
├── data.png
├── HM3D
├── MT-HM3D
├── Open_Sans
├── README.md
├── scene_init_poses_all.csv
└── scene_init_poses.csv
```

`HM3D` is decompressed from `hm3d-train-habitat-v0.2.tar` which can be downloaded [here](https://github.com/matterport/habitat-matterport-3dresearch).

`MT-HM3D` can be obtained from [here](https://huggingface.co/datasets/zmling/MT-HM3D).


## Inference on MT-HM3D
```
sh scripts/run_memory_eqa.sh
```

## Acknowledgements

Our project is built upon [MemoryEQA](https://github.com/memory-eqa/MemoryEQA) and [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa).
