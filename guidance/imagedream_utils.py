import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from imagedream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from imagedream.model_zoo import build_model
from imagedream.ldm.models.diffusion.ddim import DDIMSampler

from diffusers import DDIMScheduler

class ImageDream(nn.Module):
    def __init__(
        self,
        device,
        model_name='sd-v2.1-base-4view-ipmv',
        ckpt_path=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.model = build_model(self.model_name, ckpt_path=self.ckpt_path).eval().to(self.device)
        self.model.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dtype = torch.float32

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.image_embeddings = {}
        self.embeddings = {}

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=self.dtype
        )

    @torch.no_grad()
    def get_image_text_embeds(self, image, prompts, negative_prompts):

        image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
        image_pil = TF.to_pil_image(image[0])
        image_embeddings = self.model.get_learned_image_conditioning(image_pil).repeat(5,1,1) # [5, 257, 1280]
        self.image_embeddings['pos'] = image_embeddings
        self.image_embeddings['neg'] = torch.zeros_like(image_embeddings)

        self.image_embeddings['ip_img'] = self.encode_imgs(image)
        self.image_embeddings['neg_ip_img'] = torch.zeros_like(self.image_embeddings['ip_img'])

        pos_embeds = self.encode_text(prompts).repeat(5,1,1)
        neg_embeds = self.encode_text(negative_prompts).repeat(5,1,1)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings
    
    @torch.no_grad()
    def refine(self, pred_rgb, camera,
               guidance_scale=5, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        
        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)

        # extra view
        camera = camera.view(real_batch_size, 4, 16)
        camera = torch.cat([camera, torch.zeros_like(camera[:, :1])], dim=1) # [rB, 5, 16]
        camera = camera.view(real_batch_size * 5, 16)

        camera = camera.repeat(2, 1)
        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        image_embeddings = torch.cat([self.image_embeddings['neg'].repeat(real_batch_size, 1, 1), self.image_embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        ip_img_embeddings= torch.cat([self.image_embeddings['neg_ip_img'].repeat(real_batch_size, 1, 1, 1), self.image_embeddings['ip_img'].repeat(real_batch_size, 1, 1, 1)], dim=0)
        
        context = {
            "context": embeddings, 
            "ip": image_embeddings, 
            "ip_img": ip_img_embeddings,
            "camera": camera, 
            "num_frames": 4 + 1
        }

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):

            # extra view
            
            latents = latents.view(real_batch_size, 4, 4, 32, 32)
            latents = torch.cat([latents, torch.zeros_like(latents[:, :1])], dim=1).view(-1, 4, 32, 32)
            latent_model_input = torch.cat([latents] * 2)

            tt = torch.cat([t.unsqueeze(0).repeat(real_batch_size * 5)] * 2).to(self.device)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            
            # remove extra view
            noise_pred_uncond = noise_pred_uncond.reshape(real_batch_size, 5, 4, 32, 32)[:, :-1].reshape(-1, 4, 32, 32)
            noise_pred_cond = noise_pred_cond.reshape(real_batch_size, 5, 4, 32, 32)[:, :-1].reshape(-1, 4, 32, 32)
            latents = latents.reshape(real_batch_size, 5, 4, 32, 32)[:, :-1].reshape(-1, 4, 32, 32)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb, # [B, C, H, W]
        camera, # [B, 4, 4]
        step_ratio=None,
        guidance_scale=5,
        as_latent=False,
    ):
        
        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_256)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)

        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)

        # extra view
        camera = camera.view(real_batch_size, 4, 16)
        camera = torch.cat([camera, torch.zeros_like(camera[:, :1])], dim=1) # [rB, 5, 16]
        camera = camera.view(real_batch_size * 5, 16)

        camera = camera.repeat(2, 1)
        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        image_embeddings = torch.cat([self.image_embeddings['neg'].repeat(real_batch_size, 1, 1), self.image_embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        ip_img_embeddings= torch.cat([self.image_embeddings['neg_ip_img'].repeat(real_batch_size, 1, 1, 1), self.image_embeddings['ip_img'].repeat(real_batch_size, 1, 1, 1)], dim=0)
        
        context = {
            "context": embeddings, 
            "ip": image_embeddings, 
            "ip_img": ip_img_embeddings,
            "camera": camera, 
            "num_frames": 4 + 1
        }

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise) # [B=4, 4, 32, 32]
            # extra view
            t = t.view(real_batch_size, 4)
            t = torch.cat([t, t[:, :1]], dim=1).view(-1)
            latents_noisy = latents_noisy.view(real_batch_size, 4, 4, 32, 32)
            latents_noisy = torch.cat([latents_noisy, torch.zeros_like(latents_noisy[:, :1])], dim=1).view(-1, 4, 32, 32)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # import kiui
            # kiui.lo(latent_model_input, t, context['context'], context['camera'])
            
            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

            # remove extra view
            noise_pred_uncond = noise_pred_uncond.reshape(real_batch_size, 5, 4, 32, 32)[:, :-1].reshape(-1, 4, 32, 32)
            noise_pred_cond = noise_pred_cond.reshape(real_batch_size, 5, 4, 32, 32)[:, :-1].reshape(-1, 4, 32, 32)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        grad = (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents # [B, 4, 32, 32]

    @torch.no_grad()
    def prompt_to_img(
        self,
        image,
        prompts,
        negative_prompts="",
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=5.0,
        latents=None,
        elevation=0,
        azimuth_start=0,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        
        real_batch_size = len(prompts)
        batch_size = len(prompts) * 5

        # Text embeds -> img latents
        sampler = DDIMSampler(self.model)
        shape = [4, height // 8, width // 8]

        c_ = {"context": self.encode_text(prompts).repeat(5,1,1)}
        uc_ = {"context": self.encode_text(negative_prompts).repeat(5,1,1)}

        # image embeddings
        image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
        image_pil = TF.to_pil_image(image[0])
        image_embeddings = self.model.get_learned_image_conditioning(image_pil).repeat(5,1,1).to(self.device)
        c_["ip"] = image_embeddings
        uc_["ip"] = torch.zeros_like(image_embeddings)
            
        ip_img = self.encode_imgs(image)
        c_["ip_img"] = ip_img
        uc_["ip_img"] = torch.zeros_like(ip_img)

        camera = get_camera(4, elevation=elevation, azimuth_start=azimuth_start, extra_view=True)
        camera = camera.repeat(real_batch_size, 1).to(self.device)

        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = 5

        kiui.lo(image_embeddings, ip_img, camera)

        latents, _ = sampler.sample(S=num_inference_steps, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=guidance_scale,
                                        unconditional_conditioning=uc_,
                                        eta=0, x_T=None)

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [4, 3, 256, 256]

        kiui.lo(latents, imgs)
        
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import kiui

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--steps", type=int, default=30)
    opt = parser.parse_args()

    device = torch.device("cuda")

    sd = ImageDream(device)

    image = kiui.read_image(opt.image, mode='tensor')
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)

    while True:
        imgs = sd.prompt_to_img(image, opt.prompt, opt.negative, num_inference_steps=opt.steps)

        grid = np.concatenate([
            np.concatenate([imgs[0], imgs[1]], axis=1),
            np.concatenate([imgs[2], imgs[3]], axis=1),
        ], axis=0)

        # visualize image
        plt.imshow(grid)
        plt.show()
