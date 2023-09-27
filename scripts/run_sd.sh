export CUDA_VISIBLE_DEVICES=6

# easy samples
python main.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream 
python main2.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream 
python main.py --config configs/text.yaml prompt="a ripe strawberry" save_path=strawberry 
python main2.py --config configs/text.yaml prompt="a ripe strawberry" save_path=strawberry 
python main.py --config configs/text.yaml prompt="a blue tulip" save_path=tulip 
python main2.py --config configs/text.yaml prompt="a blue tulip" save_path=tulip 

python main.py --config configs/text.yaml prompt="a golden goblet" save_path=goblet 
python main2.py --config configs/text.yaml prompt="a golden goblet" save_path=goblet 
python main.py --config configs/text.yaml prompt="a photo of a hamburger" save_path=hamburger 
python main2.py --config configs/text.yaml prompt="a photo of a hamburger" save_path=hamburger 
python main.py --config configs/text.yaml prompt="a delicious croissant" save_path=croissant 
python main2.py --config configs/text.yaml prompt="a delicious croissant" save_path=croissant 

# hard samples
python main.py --config configs/text.yaml prompt="a baby bunny sitting on top of a stack of pancake" save_path=bunny_pancake 
python main2.py --config configs/text.yaml prompt="a baby bunny sitting on top of a stack of pancake" save_path=bunny_pancake 
python main.py --config configs/text.yaml prompt="a typewriter" save_path=typewriter 
python main2.py --config configs/text.yaml prompt="a typewriter" save_path=typewriter 
python main.py --config configs/text.yaml prompt="a pineapple" save_path=pineapple 
python main2.py --config configs/text.yaml prompt="a pineapple" save_path=pineapple 

python main.py --config configs/text.yaml prompt="a model of a house in Tudor style" save_path=tudor_house 
python main2.py --config configs/text.yaml prompt="a model of a house in Tudor style" save_path=tudor_house 
python main.py --config configs/text.yaml prompt="a lionfish" save_path=lionfish 
python main2.py --config configs/text.yaml prompt="a lionfish" save_path=lionfish 
python main.py --config configs/text.yaml prompt="a bunch of yellow rose, highly detailed" save_path=rose 
python main2.py --config configs/text.yaml prompt="a bunch of yellow rose, highly detailed" save_path=rose 
