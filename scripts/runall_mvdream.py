import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
args = parser.parse_args()

prompts = [
    # ('butterfly', 'a beautiful, intricate butterfly'),
    # ('boy', 'a nendoroid of a chibi cute boy'),
    # ('axe', 'a viking axe, fantasy, blender'),
    # ('dog_rocket', 'corgi riding a rocket'),
    ('teapot', 'a chinese teapot'),
    ('squirrel_guitar', 'a DSLR photo of a squirrel playing guitar'),
    # ('house', 'fisherman house, cute, cartoon, blender, stylized'),
    # ('ship', 'Higly detailed, majestic royal tall ship, realistic painting'),
    ('einstein', 'Albert Einstein with grey suit is riding a bicycle'),
    # ('angle', 'a statue of an angle'),
    ('lion', 'A 3D model of Simba, the lion cub from The Lion King, standing majestically on Pride Rock, character'),
    # ('paris', 'mini Paris, highly detailed 3d model'),
    # ('pig_backpack', 'a pig wearing a backpack'),
    ('pisa_tower', 'Picture of the Leaning Tower of Pisa, featuring its tilted structure and marble facade'),
    # ('robot', 'a human-like full body robot'),
    ('coin', 'a golden coin'),
    # ('cake', 'a delicious and beautiful cake'),
    # ('horse', 'a DSLR photo of a horse'),
    # ('cat', 'a photo of a cat'),
    ('cat_hat', 'a photo of a cat wearing a wizard hat'),
    # ('cat_ball', 'a photo of a cat playing with a red ball'),
    # ('nendoroid', 'a nendoroid of a chibi girl'),

]

for name, prompt in prompts:
    print(f'======== processing {name} ========')
    # first stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main.py --config configs/text_mv.yaml prompt="{prompt}" save_path={name}')
    # second stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main2.py --config configs/text_mv.yaml  prompt="{prompt}" save_path={name}')
    # export video
    mesh_path = os.path.join('logs', f'{name}.obj')
    os.makedirs('videos', exist_ok=True)
    os.system(f'python -m kiui.render {mesh_path} --save_video videos/{name}.mp4 --wogui')