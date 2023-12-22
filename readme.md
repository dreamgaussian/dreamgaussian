# DreamGaussian

This repository contains the official implementation for [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653).

### [Project Page](https://dreamgaussian.github.io) | [Arxiv](https://arxiv.org/abs/2309.16653)

https://github.com/dreamgaussian/dreamgaussian/assets/25863658/db860801-7b9c-4b30-9eb9-87330175f5c8

### News

- 2023.12.22: add experimental support for [ImageDream](https://github.com/bytedance/ImageDream), check [imagedream.yaml](./configs/image_sai.yaml).
- 2023.12.14: add support for [Stable-Zero123](https://stability.ai/news/stable-zero123-3d-generation), check [image_sai.yaml](./configs/image_sai.yaml).
- 2023.10.21: add support for [MVDream](https://github.com/bytedance/MVDream), check [text_mv.yaml](./configs/text_mv.yaml).

### [Colab demo](https://github.com/camenduru/dreamgaussian-colab)

- Image-to-3D: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sLpYmmLS209-e5eHgcuqdryFRRO6ZhFS?usp=sharing)
- Text-to-3D: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/dreamgaussian-colab/blob/main/dreamgaussian_colab.ipynb)

### [Gradio demo](https://huggingface.co/spaces/jiawei011/dreamgaussian)

- Image-to-3D: <a href="https://huggingface.co/spaces/jiawei011/dreamgaussian"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>
- Run Gradio demo on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1owXJthskHoVXBNvxUB0Bg0JP2Rc7QsTe?usp=sharing)

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

# To use MVdream, also install:
pip install git+https://github.com/bytedance/MVDream

# To use ImageDream, also install:
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream
```

Tested on:

- Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.
- Windows 10 with torch 2.1 & CUDA 12.1 on a 3070.

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
# auto load coarse_mesh and refine 50 iters (~1min), export fine_mesh to logs
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name

# specify coarse mesh path explicity
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name mesh=logs/name_mesh.obj

# gui mode
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name gui=True

# export glb instead of obj
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name mesh_format=glb

### visualization
# gui for visualizing mesh
# `kire` is short for `python -m kiui.render`
kire logs/name.obj

# save 360 degree video of mesh (can run without gui)
kire logs/name.obj --save_video name.mp4 --wogui

# save 8 view images of mesh (can run without gui)
kire logs/name.obj --save images/name/ --wogui

### evaluation of CLIP-similarity
python -m kiui.cli.clip_sim data/name_rgba.png logs/name.obj
```

Please check `./configs/image.yaml` for more options.

Image-to-3D (stable-zero123):

```bash
### training gaussian stage
python main.py --config configs/image_sai.yaml input=data/name_rgba.png save_path=name

### training mesh stage
python main2.py --config configs/image_sai.yaml input=data/name_rgba.png save_path=name
```

Text-to-3D:

```bash
### training gaussian stage
python main.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream

### training mesh stage
python main2.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream
```

Please check `./configs/text.yaml` for more options.

Text-to-3D (MVDream):

```bash
### training gaussian stage
python main.py --config configs/text_mv.yaml prompt="a plush toy of a corgi nurse" save_path=corgi_nurse

### training mesh stage
python main2.py --config configs/text_mv.yaml prompt="a plush toy of a corgi nurse" save_path=corgi_nurse
```

Please check `./configs/text_mv.yaml` for more options.

Image+Text-to-3D (ImageDream):

```bash
### training gaussian stage
python main.py --config configs/imagedream.yaml input=data/ghost_rgba.png prompt="a ghost eating hamburger" save_path=ghost

### training mesh stage
python main2.py --config configs/imagedream.yaml input=data/ghost_rgba.png prompt="a ghost eating hamburger" save_path=ghost
```

Helper scripts:

```bash
# run all image samples (*_rgba.png) in ./data
python scripts/runall.py --dir ./data --gpu 0

# run all text samples (hardcoded in runall_sd.py)
python scripts/runall_sd.py --gpu 0

# export all ./logs/*.obj to mp4 in ./videos
python scripts/convert_obj_to_video.py --dir ./logs
```

Gradio Demo:

```bash
python gradio_app.py
```

## Tips
* The world & camera coordinate system is the same as OpenGL:
```
    World            Camera        
  
     +y              up  target                                              
     |               |  /                                            
     |               | /                                                
     |______+x       |/______right                                      
    /                /         
   /                /          
  /                /           
 +z               forward           

elevation: in (-90, 90), from +y to -y is (-90, 90)
azimuth: in (-180, 180), from +z to +x is (0, 90)
```

* Trouble shooting OpenGL errors (e.g., `[F glutil.cpp:338] eglInitialize() failed`): 
```bash
# either try to install OpenGL correctly (usually installed with the Nvidia driver), or use force_cuda_rast:
python main.py --config configs/image_sai.yaml input=data/name_rgba.png save_path=name force_cuda_rast=True

kire mesh.obj --force_cuda_rast
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

## Citation

```
@article{tang2023dreamgaussian,
  title={DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation},
  author={Tang, Jiaxiang and Ren, Jiawei and Zhou, Hang and Liu, Ziwei and Zeng, Gang},
  journal={arXiv preprint arXiv:2309.16653},
  year={2023}
}
```
