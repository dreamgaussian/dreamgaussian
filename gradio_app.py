import gradio as gr
import os
from PIL import Image
import subprocess


# check if there is a picture uploaded or selected
def check_img_input(control_image):
    if control_image is None:
        raise gr.Error("Please select or upload an input image")


def optimize_stage_1(image_block: Image.Image, preprocess_chk: bool, elevation_slider: float):
    if not os.path.exists('tmp_data'):
        os.makedirs('tmp_data')
    if preprocess_chk:
        # save image to a designated path
        image_block.save(os.path.join('tmp_data', 'tmp.png'))

        # preprocess image
        print(f'python process.py {os.path.join("tmp_data", "tmp.png")}')
        subprocess.run(f'python process.py {os.path.join("tmp_data", "tmp.png")}', shell=True)
    else:
        image_block.save(os.path.join('tmp_data', 'tmp_rgba.png'))

    # stage 1
    subprocess.run(f'python main.py --config {os.path.join("configs", "image.yaml")} input={os.path.join("tmp_data", "tmp_rgba.png")} save_path=tmp mesh_format=glb elevation={elevation_slider} force_cuda_rast=True', shell=True)

    return os.path.join('logs', 'tmp_mesh.glb')


def optimize_stage_2(elevation_slider: float):
    # stage 2
    subprocess.run(f'python main2.py --config {os.path.join("configs", "image.yaml")} input={os.path.join("tmp_data", "tmp_rgba.png")} save_path=tmp mesh_format=glb elevation={elevation_slider} force_cuda_rast=True', shell=True)

    return os.path.join('logs', 'tmp.glb')


if __name__ == "__main__":
    _TITLE = '''DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation'''

    _DESCRIPTION = '''
    <div>
    <a style="display:inline-block" href="https://dreamgaussian.github.io"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
    <a style="display:inline-block; margin-left: .5em" href="https://arxiv.org/abs/2309.16653"><img src="https://img.shields.io/badge/2309.16653-f9f7f7?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAABMCAYAAADJPi9EAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAuIwAALiMBeKU/dgAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAa2SURBVHja3Zt7bBRFGMAXUCDGF4rY7m7bAwuhlggKStFgLBgFEkCIIRJEEoOBYHwRFYKilUgEReVNJEGCJJpehHI3M9vZvd3bUP1DjNhEIRQQsQgSHiJgQZ5dv7krWEvvdmZ7d7vHJN+ft/f99pv5XvOtJMFCqvoCUpTdIEeRLC+L9Ox5i3Q9LACaCeK0kXoSChVcD3C/tQPHpAEsquQ73IkUcEz2kcLCknyGW5MGjkljRFVL8xJOKyi4CwCOuQAeAkfTP1+tNxLkogvgEbDgffkJqKqvuMA5ifOpqg/5qWecRstNg7xoUTI1Fovdxg8oy2s5AP8CGeYHmGngeZaOL4I4LXLcpHg4149/GDz4xqgsb+UAbMKKUpkrqHA43MUyyJpWUK0EHeG2YKRXr7tB+QMcgGewLD+ebTDbtrtbBt7UPlhS4rV4IvcDI7J8P1OeA/AcAI7LHljN7aB8XTowJmZt9EFRD/o0SDMH4HlwMhMyDWZZSAHFf3YDs3RS49WDLuaAY3IJq+qzmQKLxXAZKN7oDoYbdV3v5elPqiSpMyiOuAEVZVqHXb1OhloUH+MA+ztO0cAO/RkrfyBE7OAEbAZvO8vzVtTRWFD6DAfY5biBM3PWiaL0a4lvXICwnV8WjmE6ntYmhqX2jjp5LbMZjCw/wbYeN6CizOa2GMVzQOlmHjB4Ceuyk6LJ8huccEmR5Xddg7OOV/NAtchW+E3XbOag60QA4Qwuarca0bRuEJyr+cFQwzcY98huxhAKdQelt4kAQpj4qJ3gvFXAYn+aJumXk1yPlpQUgtIHhbYoFMUstNRRWgjnpl4A7IKlayNymqFHFaWCpV9CFry3LGxR1CgA5kB5M8OX2goApwpaz6mdOMGxtAgXWJySxb4WuQD4qTDgU+N5AAnzpr7ChSWpCyisiQJqY0Y7FtmSKpbV23b45kC0KHBxcQ9QeI8w4KgnHRPVtIU7rOtbioLVg5Hl/qDwSVFAMqLSMSObroCdZYlzIJtMRFVHCaRo/wFWPgaAXzdbBpkc2A4aKzCNd97+URQuESYGDDhIVfWOQIKZJu4D2+oXlgDTV1865gUQZDts756BArMNMoR1oa46BYqbyPixZz1ZUFV3sgwoGBajuBKATl3btIn8QYYMuezRgrsiRUWyr2BxA40EkPMpA/Hm6gbUu7fjEXA3azP6AsbKD9bxdUuhjM9W7fII52BF+daRpE4+WA3P501+jbfmHvQKyFqMuXf7Ot4mkN2fr50y+bRH61X7AXdUpHSxaPQ4GVbR5AGw3g+434XgQGKfr72I+vQRhfsu92dOx7WicInzt3CBg1RVpMm0NveWo2SqFzgmdNZMbriILD+S+zoueWf2vSdAipzacWN5nMl6XxNlUHa/J8DoJodUDE0HR8Ll5V0lPxcrLEHZPV4AzS83OLis7FowVa3RSku7BSNxJqQAlN3hBTC2apmDSkpaw22wJemGQFUG7J4MlP3JC6A+f96V7vRyX9It3nzT/GrjIU8edM7rMSnIi10f476lzbE1K7yEiEuWro0OJBguLCwDuFOJc1Na6sRWL/cCeMIwUN9ggSVbe3v/5/EgzTKWLvEAiBrYRUkgwNI2ZaFQNT75UDxEUEx97zYnzpmiLEmbaYCbNxYtFAb0/Z4AztgUrhyxuNgxPnhfHFDHz/vTgFWUQZxTRkkJhQ6YNdVUEPAfO6ZV5BRss6LcCVb7VaAma9giy0XJZBt9IQh42NY0NSdgbLIPlLUF6rEdrdt0CUCK1wsCbkcI3ZSLc7ZSwGLbmJXbPsNxnE5xilYKAobZ77LpGZ8TAIun+/iCKQoF71IxQDI3K2CCd+ARNvXg9sykBcnHAoCZG4u66hlDoQLe6QV4CRtFSxZQ+D0BwNO2jgdkzoGoah1nj3FVlSR19taTSYxI8QLut23U8dsgzqHulJNCQpcqBnpTALCuQ6NSYLHpmR5i42gZzuIdcrMMvMJbQlxe3jXxyZnLACl7ARm/FjPIDOY8ODtpM71sxwfcZpvBeUzKWmfNINM5AS+wO0Khh7dMqKccu4+qatarZjYAwDlgetzStHtEt+XedsBOQtU9XMrRgjg4KTnc5nr+dmqadit/4C4uLm8DuA9koJTj1TL7fI5nDL+qqoo/FLGAzL7dYT17PzvAcQONYSUQRxW/QMrHZVIyik0ZuQA2mzp+Ji8BW4YM3Mbzm9inaHkJCGfrUZZjujiYailfFwA8DHIy3acwUj4v9vUVa+SmgNsl5fuyDTKovW9/IAmfLV0Pi2UncA515kjYdrwC9i9rpuHiq3JwtAAAAABJRU5ErkJggg=="></a>
    <a style="display:inline-block; margin-left: .5em" href='https://github.com/dreamgaussian/dreamgaussian'><img src='https://img.shields.io/github/stars/dreamgaussian/dreamgaussian?style=social'/></a>
    </div>
    We present DreamGausssion, a 3D content generation framework that significantly improves the efficiency of 3D content creation. 
    '''
    _IMG_USER_GUIDE = "Please upload an image in the block above (or choose an example above) and click **Generate 3D**."

    # load images in 'data' folder as examples
    example_folder = os.path.join(os.path.dirname(__file__), 'data')
    example_fns = os.listdir(example_folder)
    example_fns.sort()
    examples_full = [os.path.join(example_folder, x) for x in example_fns if x.endswith('.png')]

    # Compose demo layout & data flow
    with gr.Blocks(title=_TITLE, theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        # Image-to-3D
        with gr.Row(variant='panel'):
            with gr.Column(scale=5):
                image_block = gr.Image(type='pil', image_mode='RGBA', height=290, label='Input image', tool=None)

                elevation_slider = gr.Slider(-90, 90, value=0, step=1, label='Estimated elevation angle')
                gr.Markdown(
                    "default to 0 (horizontal), range from [-90, 90]. If you upload a look-down image, try a value like -30")

                preprocess_chk = gr.Checkbox(True,
                                             label='Preprocess image automatically (remove background and recenter object)')

                gr.Examples(
                    examples=examples_full,  # NOTE: elements must match inputs list!
                    inputs=[image_block],
                    outputs=[image_block],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=40
                )
                img_run_btn = gr.Button("Generate 3D")
                img_guide_text = gr.Markdown(_IMG_USER_GUIDE, visible=True)

            with gr.Column(scale=5):
                obj3d_stage1 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model (Stage 1)")
                obj3d = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model (Final)")

            # if there is an input image, continue with inference
            # else display an error message
            img_run_btn.click(check_img_input, inputs=[image_block], queue=False).success(optimize_stage_1,
                                                                                          inputs=[image_block,
                                                                                                  preprocess_chk,
                                                                                                  elevation_slider],
                                                                                          outputs=[
                                                                                              obj3d_stage1]).success(
                optimize_stage_2, inputs=[elevation_slider], outputs=[obj3d])

    demo.queue().launch(share=True)