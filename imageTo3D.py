import sys
import pathlib
import subprocess

# Simple script to process image, run main.py, run main2.py
# and save everything into a folder

def run_main(inputImagePath, outputDirName, imageSize):
    inputImagePath = pathlib.Path(inputImagePath)
    img_path = inputImagePath.parent / (inputImagePath.stem + "_rgba.png")
    print("Processed image path =", img_path)

    process = subprocess.Popen(["python", "process.py", inputImagePath])
    process.wait()
    print("--- process.py finished ---")

    print("Running main.py...")
    main1 = subprocess.Popen(["python", "main.py", "--config", "configs/image.yaml",
                                f"input={img_path}", f"save_path={outputDirName}\{outputDirName}", "--size", str(imageSize)])  # Fix the argument format
    main1.wait()
    print("--- main.py finished ---")

    print("Running main2.py...")
    main2 = subprocess.Popen(["python", "main2.py", "--config", "configs/image.yaml",
                                f"input={img_path}", f"save_path={outputDirName}\{outputDirName}", "--size", str(imageSize)])  # Fix the script name and argument format
    main2.wait()
    print("--- main2.py finished ---")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python imageTo3D.py <input_image_path> <output_directory_name> <image_size>")
        sys.exit(1)

    inputImagePath = sys.argv[1]
    outputDirName = sys.argv[2]
    imageSize = sys.argv[3]
    
    run_main(inputImagePath, outputDirName, imageSize)