import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='data', type=str, help='Directory where processed images are stored')
parser.add_argument('--out', default='logs', type=str, help='Directory where obj files will be saved')
parser.add_argument('--video-out', default='videos', type=str, help='Directory where videos will be saved')
parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
parser.add_argument('--elevation', default=0, type=int, help='Elevation angle of view in degrees')
parser.add_argument('--config', default='configs', type=str, help='Path to config directory, which contains image.yaml')
args = parser.parse_args()

files = glob.glob(f'{args.dir}/*_rgba.png')
configs_dir = args.config

# check if image.yaml exists
if not os.path.exists(os.path.join(configs_dir, 'image.yaml')):
    raise FileNotFoundError(
        f'image.yaml not found in {configs_dir} directory. Please check if the directory is correct.'
    )

# create output directories if not exists
out_dir = args.out
os.makedirs(out_dir, exist_ok=True)
video_dir = args.video_out
os.makedirs(video_dir, exist_ok=True)


for file in files:
    name = os.path.basename(file).replace("_rgba.png", "")
    print(f'======== processing {name} ========')
    # first stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main.py '
              f'--config {configs_dir}/image.yaml '
              f'input={file} '
              f'save_path={name} elevation={args.elevation}')
    # second stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main2.py '
              f'--config {configs_dir}/image.yaml '
              f'input={file} '
              f'save_path={name} elevation={args.elevation}')
    # export video
    mesh_path = os.path.join(out_dir, f'{name}.obj')
    os.system(f'python -m kiui.render {mesh_path} '
              f'--save_video {video_dir}/{name}.mp4 '
              f'--wogui '
              f'--elevation {args.elevation}')
