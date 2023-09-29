import os
import math
import cv2
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from mesh import Mesh, safe_normalize

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)

class Renderer(nn.Module):
    def __init__(self, opt):
        
        super().__init__()

        self.opt = opt

        self.mesh = Mesh.load(self.opt.mesh, resize=False)

        if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()
        
        # extract trainable parameters
        self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v))
        self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(self.mesh.albedo))


    def get_params(self):

        params = [
            {'params': self.raw_albedo, 'lr': self.opt.texture_lr},
        ]

        if self.opt.train_geo:
            params.append({'params': self.v_offsets, 'lr': self.opt.geom_lr})

        return params

    @torch.no_grad()
    def export_mesh(self, save_path):
        self.mesh.v = (self.mesh.v + self.v_offsets).detach()
        self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
        self.mesh.write(save_path)

    
    def render(self, pose, proj, h0, w0, ssaa=1, bg_color=1, texture_filter='linear-mipmap-linear'):
        
        # do super-sampling
        if ssaa != 1:
            h = make_divisible(h0 * ssaa, 8)
            w = make_divisible(w0 * ssaa, 8)
        else:
            h, w = h0, w0
        
        results = {}

        # get v
        if self.opt.train_geo:
            v = self.mesh.v + self.v_offsets # [N, 3]
        else:
            v = self.mesh.v

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))

        alpha = (rast[0, ..., 3:] > 0).float()
        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]

        texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
        albedo = dr.texture(self.raw_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode=texture_filter) # [1, H, W, 3]
        albedo = torch.sigmoid(albedo)
        # get vn and render normal
        if self.opt.train_geo:
            i0, i1, i2 = self.mesh.f[:, 0].long(), self.mesh.f[:, 1].long(), self.mesh.f[:, 2].long()
            v0, v1, v2 = v[i0, :], v[i1, :], v[i2, :]

            face_normals = torch.cross(v1 - v0, v2 - v0)
            face_normals = safe_normalize(face_normals)
            
            vn = torch.zeros_like(v)
            vn.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
            vn.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
            vn.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

            vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
        else:
            vn = self.mesh.vn
        
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        # antialias
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
        albedo = alpha * albedo + (1 - alpha) * bg_color

        # ssaa
        if ssaa != 1:
            albedo = scale_img_hwc(albedo, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            normal = scale_img_hwc(normal, (h0, w0))
            viewcos = scale_img_hwc(viewcos, (h0, w0))

        results['image'] = albedo.clamp(0, 1)
        results['alpha'] = alpha
        results['depth'] = depth
        results['normal'] = (normal + 1) / 2
        results['viewcos'] = viewcos

        return results