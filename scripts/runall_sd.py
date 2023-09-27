import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
args = parser.parse_args()

prompts = [
    ('strawberry', 'a ripe strawberry'),
    ('cactus_pot', 'a small saguaro cactus planted in a clay pot'),
    ('hamburger', 'a delicious hamburger'),
    ('icecream', 'an icecream'),
    ('tulip', 'a blue tulip'),
    ('pineapple', 'a ripe pineapple'),
    ('goblet', 'a golden goblet'),
    # ('squitopus', 'a squirrel-octopus hybrid'),
    # ('astronaut', 'Michelangelo style statue of an astronaut'),
    # ('teddy_bear', 'a teddy bear'),
    # ('corgi_nurse', 'a plush toy of a corgi nurse'),
    # ('teapot', 'a blue and white porcelain teapot'),
    # ('skull', "a human skull"),
    # ('penguin', 'a penguin'),
    # ('campfire', 'a campfire'),
    # ('donut', 'a donut with pink icing'),
    # ('cupcake', 'a birthday cupcake'),
    # ('pie', 'shepherds pie'),
    # ('cone', 'a traffic cone'),
    # ('schoolbus', 'a schoolbus'),
    # ('avocado_chair', 'a chair that looks like an avocado'),
    # ('glasses', 'a pair of sunglasses')
    # ('potion', 'a bottle of green potion'),
    # ('chalice', 'a delicate chalice'),
]

for name, prompt in prompts:
    print(f'======== processing {name} ========')
    # first stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main.py --config configs/text.yaml prompt="{prompt}" save_path={name}')
    # second stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main2.py --config configs/text.yaml  prompt="{prompt}" save_path={name}')
    # export video
    mesh_path = os.path.join('logs', f'{name}.obj')
    os.makedirs('videos', exist_ok=True)
    os.system(f'python -m kiui.render {mesh_path} --save_video videos/{name}.mp4 --wogui')