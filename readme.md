# DreamGaussian

This repository contains the official implementation for [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653).

### [Project Page](https://dreamgaussian.github.io) | [Arxiv](https://arxiv.org/abs/2309.16653)


https://github.com/dreamgaussian/dreamgaussian/assets/25863658/db860801-7b9c-4b30-9eb9-87330175f5c8

### [Colab demo](https://github.com/camenduru/dreamgaussian-colab)
* Image-to-3D: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sLpYmmLS209-e5eHgcuqdryFRRO6ZhFS?usp=sharing)
* Text-to-3D: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/dreamgaussian-colab/blob/main/dreamgaussian_colab.ipynb)


## Install
```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit
```

Tested on:
* Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.
* Windows 10 with torch 2.1 & CUDA 12.1 on a 3070.

## Usage

Image-to-3D:
```bash
### preprocess
# background removal and recentering, save rgba at 256x256
python process.py data/name.jpg

# save at a larger resolution
python process.py data/name.jpg --size 512

# process all jpg images under a dir
python process.py data

### training gaussian stage
# train 500 iters (~1min) and export ckpt & coarse_mesh to logs
python main.py --config configs/image.yaml input=data/name_rgba.png save_path=name

# gui mode (supports visualizing training)
python main.py --config configs/image.yaml input=data/name_rgba.png save_path=name gui=True

# load and visualize a saved ckpt
python main.py --config configs/image.yaml load=logs/name_model.ply gui=True

# use an estimated elevation angle if image is not front-view (e.g., common looking-down image can use -30)
python main.py --config configs/image.yaml input=data/name_rgba.png save_path=name elevation=-30

### training mesh stage
# auto load coarse_mesh.obj and refine 50 iters (~1min), export fine_mesh to logs
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name

# specify coarse mesh path explicity
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name mesh=logs/name_mesh.obj

# gui mode
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name gui=True

### visualization
# gui for visualizing mesh
python -m kiui.render logs/name.obj

# save 360 degree video of mesh (can run without gui)
python -m kiui.render logs/name.obj --save_video name.mp4 --wogui

# save 8 view images of mesh (can run without gui)
python -m kiui.render logs/name.obj --save images/name/ --wogui

### evaluation of CLIP-similarity
python -m kiui.cli.clip_sim data/name_rgba.png logs/name.obj
```
Please check `./configs/image.yaml` for more options.

Text-to-3D:
```bash
### training gaussian stage
python main.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream

### training mesh stage
python main2.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream
```
Please check `./configs/text.yaml` for more options.

Helper scripts:
```bash
# run all image samples (*_rgba.png) in ./data
python scripts/runall.py --dir ./data --gpu 0

# run all text samples (hardcoded in runall_sd.py)
python scripts/runall_sd.py --gpu 0

# export all ./logs/*.obj to mp4 in ./videos
python scripts/convert_obj_to_video.py --dir ./logs
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

* [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
* [threestudio](https://github.com/threestudio-project/threestudio)
* [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
* [dearpygui](https://github.com/hoffstadt/DearPyGui)

## Citation

```
@article{tang2023dreamgaussian,
  title={DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation},
  author={Tang, Jiaxiang and Ren, Jiawei and Zhou, Hang and Liu, Ziwei and Zeng, Gang},
  journal={arXiv preprint arXiv:2309.16653},
  year={2023}
}
```
