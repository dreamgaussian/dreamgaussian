# DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

**Authors:** Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, Gang Zeng

DreamGaussian is a groundbreaking framework for efficient 3D content creation, designed to combine quality and speed in 3D generation tasks. Unlike traditional optimization-based approaches, DreamGaussian introduces a novel 3D Gaussian Splatting model with companioned mesh extraction and texture refinement in UV space. This innovative approach significantly accelerates the convergence of 3D generative tasks, producing high-quality textured meshes from single-view images in just 2 minutes, a remarkable 10-fold acceleration compared to existing methods.

For more information visit the following pages:
[![Project Page](https://github.com/dreamgaussian/dreamgaussian/assets/25863658/db860801-7b9c-4b30-9eb9-87330175f5c8)](https://dreamgaussian.github.io) | [![Arxiv](https://arxiv.org/abs/2309.16653)](https://arxiv.org/abs/2309.16653)

---

![DreamGaussian Demo](https://github.com/dreamgaussian/dreamgaussian/assets/25863658/db860801-7b9c-4b30-9eb9-87330175f5c8)

## Try it on Colab

- **Image-to-3D**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sLpYmmLS209-e5eHgcuqdryFRRO6ZhFS?usp=sharing)
- **Text-to-3D**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/dreamgaussian-colab/blob/main/dreamgaussian_colab.ipynb)


## Install

```bash
# pip: This is the command-line tool for installing Python packages. It allows you to easily install, upgrade, and manage Python packages from the Python Package Index (PyPI) and other sources.

pip install -r requirements.txt

# Install a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Install simple-knn
# Designed to perform operations related to k-nearest neighbors (KNN) calculations on 3D points. It includes functions for calculating Morton codes for spatial indexing, finding the minimum and maximum values among 3D points, sorting indices based on Morton codes, and computing mean distances for KNN calculations. The code is likely used for efficient nearest neighbor search in a 3D point cloud or similar data structure.
pip install ./simple-knn

# Install nvdiffrast
#Nvdiffrast is a library compatible with PyTorch and TensorFlow, offering high-performance operations for differentiable rendering based on rasterization techniques.
pip install git+https://github.com/NVlabs/nvdiffrast/

# Install kiuikit
#Kiuikit is a collection of maintained, reusable, and trustworthy code snippets designed for personal use. It includes features like lazy imports to prevent code slowdown and offers useful command-line tools, including a GUI mesh renderer.
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
#Purpose: This script is used to process and run all image samples in the specified directory (./data) through the DreamGaussian model.
python scripts/runall.py --dir ./data --gpu 0

# run all text samples (hardcoded in runall_sd.py)
#Purpose: This script is used to process and run all text samples, which are hardcoded in the runall_sd.py script, through the DreamGaussian model.
python scripts/runall_sd.py --gpu 0

# export all ./logs/*.obj to mp4 in ./videos
#Purpose: This script is used to convert 3D mesh files (.obj format) located in the ./logs/ directory into video files (.mp4) located in the ./videos/ directory.
python scripts/convert_obj_to_video.py --dir ./logs
```

## Contribution Guidelines

We welcome contributions to the DreamGaussian project from the community. Before contributing, please take a moment to review the following guidelines to ensure a smooth and collaborative experience:

### Code of Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) to create a respectful and inclusive environment for everyone in the community.

### How to Contribute

1. **Fork the Repository:** Click the "Fork" button on the top right-hand corner of this repository to create your own fork.

2. **Clone Your Fork:** Clone your forked repository to your local machine using the following command:

3. **Create a Branch:** Create a new branch for your contribution. A descriptive branch name is encouraged:

4. **Make Changes:** Make your changes or improvements to the codebase. Ensure that your code follows our coding conventions and standards.

5. **Commit Changes:** Commit your changes with clear and concise commit messages:

6. **Push to Your Fork:** Push your changes to your forked repository:

7. **Open a Pull Request:** Go to the [original repository](https://github.com/dreamgaussian/dreamgaussian) and open a pull request. Provide a clear title and description of your changes.

### Code Contribution Rules

When contributing code to DreamGaussian, please follow these essential rules:

- Adhere to the project's coding style and formatting guidelines.
- Write clear and concise code comments and documentation.
- Ensure that your code is well-tested, and write appropriate test cases.
- Keep pull requests focused on a single task or feature.
- Avoid making unrelated code changes in a single pull request.
- Before submitting a pull request, ensure your changes do not break existing functionality.
- Be responsive to feedback and comments on your pull requests.

### Reporting Issues

If you find any bugs, issues, or have suggestions for improvements, please open a [new issue](https://github.com/dreamgaussian/dreamgaussian/issues) with a detailed description of the problem or enhancement request.

Thank you for considering contributing to DreamGaussian!

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
