export CUDA_VISIBLE_DEVICES=5

# image-to-3d
python main.py --config configs/image.yaml input=data/anya_rgba.png save_path=anya
python main2.py --config configs/image.yaml input=data/anya_rgba.png save_path=anya
python -m kiui.render logs/anya.obj --save_video videos/anya.mp4 --wogui

# text-to-3d
python main.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream 
python main2.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream 
python -m kiui.render logs/icecream.obj --save_video videos/icecream.mp4 --wogui

# text-to-3d (mv)
python main.py --config configs/text_mv.yaml prompt="a photo of a sleeping cat" save_path=sleep_cat 
python main2.py --config configs/text_mv.yaml prompt="a photo of a sleeping cat" save_path=sleep_cat 
python -m kiui.render logs/sleep_cat.obj --save_video videos/sleep_cat.mp4 --wogui