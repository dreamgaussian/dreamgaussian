import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='data', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--elevation', default=0, type=int)
args = parser.parse_args()

files = glob.glob(f'{args.dir}/*_rgba.png')
for file in files:
    name = os.path.basename(file).replace("_rgba.png", "")
    print(f'======== processing {name} ========')
    # first stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main.py --config configs/image.yaml input={file} save_path={name} elevation={args.elevation}')
    # second stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main2.py --config configs/image.yaml input={file} save_path={name} elevation={args.elevation}')
    # export video
    mesh_path = os.path.join('logs', f'{name}.obj')
    os.makedirs('videos', exist_ok=True)
    os.system(f'python -m kiui.render {mesh_path} --save_video videos/{name}.mp4 --wogui --elevation {args.elevation}')