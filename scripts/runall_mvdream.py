import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
args = parser.parse_args()

prompts = [
    ('lamp', 'lamp, zen, realistic, 8K, HDR'),
    ('snowman', 'a cute snowman'),
    ('crystal', 'large crystal, incriptions, pixar style, pastel colors'),
    ('dinosour', 'cute dinosaur, pixar style'),
    ('owl', 'a cute owl, low poly, pbr'),
    ('rocket', 'knitted rocket, knitted, embroidery details, photorealistic, 4k, plushie, HD, detailed'),
    ('rainbow_octopus', 'rainbow octopus plush toy'),
    ('chicken', 'low poly adorable chicken'),
    ('cathoodie', 'a cat in a hoodie'),
    ('catfrog', 'a cat frog'),
    ('lantern', 'a mystical lantern'),
    # ('squirrel', 'a DSLR photo of a squirrel.'),
    # ('corgi', 'a DSLR photo of a corgi dog.'),
    # ('motorcycles', 'a DSLR photo of a motorcycles.'),
    # ('elephant', 'a DSLR photo of an elephant.'),
    # ('cat_cup', 'a kitten in a mug cup.'),
    # ('fox', 'a DSLR photo of a red fox.'),
    # ('zebra', 'a zebra'),
    # ('butterfly', 'a beautiful, intricate butterfly'),
    # ('boy', 'a nendoroid of a chibi cute boy'),
    # ('dagger', 'stylized, dark, psychedelic, neon ceremonial dagger'),
    # ('axe', 'a viking axe, fantasy, blender'),
    # ('dog_rocket', 'corgi riding a rocket'),
    # ('teapot', 'a chinese teapot'),
    # ('house', 'fisherman house, cute, cartoon, blender, stylized'),
    # ('ship', 'Higly detailed, majestic royal tall ship, realistic painting'),
    # ('einstein', 'Albert Einstein with grey suit is riding a bicycle'),
    # ('angle', 'a statue of an angle'),
    # ('lion', 'A 3D model of Simba, the lion cub from The Lion King, standing majestically on Pride Rock, character'),
    # ('paris', 'mini Paris, highly detailed 3d model'),
    # ('pig_backpack', 'a pig wearing a backpack'),
    # ('pisa_tower', 'Picture of the Leaning Tower of Pisa, featuring its tilted structure and marble facade'),
    # ('robot', 'a human-like full body robot'),
    # ('astronaut', 'an astronaut, full body'),
    # ('starfruit', 'a DLSR photo of a starfruit'),
    # ('pineapple', 'a DLSR photo of a pineapple'),
    # ('cake', 'a delicious and beautiful cake')
    # ('cat', 'an orange cat'),
    # ('eagle', 'an eagle'),
    # ('bike', 'a cool bike'),
    # ('book', 'a delicate and old magic book'),
    # ('cat_hat', 'a photo of a cat wearing a wizard hat'),
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