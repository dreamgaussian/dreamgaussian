import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement

import torch
from torch import nn

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distCUDA2

from sh_utils import eval_sh, SH2RGB, RGB2SH
from mesh import Mesh
from mesh_utils import decimate_mesh, clean_mesh

import kiui

def inverse_sigmoid(x):
    """Calculate the inverse of the sigmoid function.

    Parameters:
        x (float): The input value.

    Returns:
        float: The result of the inverse sigmoid function.
    """
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """Calculate a learning rate function based on the provided parameters.

    Parameters:
        lr_init (float): The initial learning rate.
        lr_final (float): The final learning rate.
        lr_delay_steps (int, optional): The number of delay steps. Defaults to 0.
        lr_delay_mult (float, optional): The delay multiplier. Defaults to 1.0.
        max_steps (int, optional): The maximum number of steps. Defaults to 1000000.

    Returns:
        function: A helper function that calculates the learning rate based on the step.
    """

    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    """
    Copy the lower diagonal elements of the given tensor to the first 6 columns of a new tensor.

    Parameters:
        L (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The new tensor with the lower diagonal elements copied.
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    """
    Helper function that calls strip_lowerdiag with the same input.

    Parameters:
        sym: The input tensor.

    Returns:
        torch.Tensor: The result of strip_lowerdiag.
    """
    return strip_lowerdiag(sym)

def gaussian_3d_coeff(xyzs, covs):
    """
    Compute the power values for each element of xyzs using covs.
    Apply a threshold to set any positive power values to -1e10.
    Return the exponential of the power values as a tensor.

    Parameters:
        xyzs (torch.Tensor): The input tensor xyzs.
        covs (torch.Tensor): The input tensor covs.

    Returns:
        torch.Tensor: The result tensor.
    """
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0

    return torch.exp(power)

def build_rotation(r):
    """
    Build a rotation matrix using the elements of the input tensor r.

    Parameters:
        r (torch.Tensor): A tensor containing the elements needed to build the rotation matrix.
            The tensor should have shape (N, 4), where N is the number of rotation matrices to compute.

    Returns:
        torch.Tensor: A tensor containing the rotation matrices. The tensor has shape (N, 3, 3).
    """
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    """
    Build a scaling and rotation matrix using the elements of the input tensors s and r.

    Parameters:
        s (torch.Tensor): A tensor containing the scaling factors. The tensor should have shape (N, 3),
            where N is the number of scaling and rotation matrices to compute.
        r (torch.Tensor): A tensor containing the elements needed to build the rotation matrix.
            The tensor should have shape (N, 4).

    Returns:
        torch.Tensor: A tensor containing the scaling and rotation matrices. The tensor has shape (N, 3, 3).
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class BasicPointCloud(NamedTuple):
    """
    A named tuple representing a basic point cloud.

    Attributes:
        points (np.array): An array of points.
        colors (np.array): An array of colors.
        normals (np.array): An array of normals.
    """
    points: np.array
    colors: np.array
    normals: np.array


class GaussianModel:

    def setup_functions(self):
        """
        Set up various activation functions.

        This method sets up the following activation functions:
        - scaling_activation: exponential function
        - scaling_inverse_activation: logarithm function
        - covariance_activation: function to build covariance from scaling and rotation
        - opacity_activation: sigmoid function
        - inverse_opacity_activation: inverse sigmoid function
        - rotation_activation: normalization function
        """
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        """
        Initialize a GaussianModel instance.

        This method initializes the following attributes:
        - active_sh_degree: 0
        - max_sh_degree: sh_degree
        - _xyz: empty tensor
        - _features_dc: empty tensor
        - _features_rest: empty tensor
        - _scaling: empty tensor
        - _rotation: empty tensor
        - _opacity: empty tensor
        - max_radii2D: empty tensor
        - xyz_gradient_accum: empty tensor
        - denom: empty tensor
        - optimizer: None
        - percent_dense: 0
        - spatial_lr_scale: 0

        Parameters:
            sh_degree (int): The SH degree
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)


    @property
    def get_opacity(self):
        """
        Get the opacity value.

        This method returns the result of calling the 'opacity_activation' function
        with the '_opacity' attribute as an argument.

        Returns:
            The opacity value.
        """
        return self.opacity_activation(self._opacity)

    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        """Extract fields from the input data.

        This function takes in several parameters and returns a tensor 'occ' representing the extracted fields. The function performs operations on the input data to extract and process fields. It uses nested loops to iterate over the X, Y, and Z dimensions of the input tensor, and within each loop, it calculates the values of xx, yy, and zz tensors using torch.meshgrid. It then concatenates these tensors and applies some calculations to obtain a tensor 'pts'. The function then applies some operations on 'pts' and other tensors to calculate the final value of 'occ'.

        Parameters:
            resolution (int): The resolution of the field.
            num_blocks (int): The number of blocks.
            relax_ratio (float): The relaxation ratio.

        Returns:
            torch.Tensor: The extracted fields.
        """
        # resolution: resolution of field

        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]

        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)

                    # kiui.lo(val, mask_opas, w)

                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 

        kiui.lo(occ, verbose=1)

        return occ

    def extract_mesh(self, path, density_thresh=1, resolution=128, decimate_target=1e5):
        """
        Extracts a mesh from the given path.

        This method creates a mesh by extracting fields from the given resolution and applying a density threshold. The resulting mesh is then transformed and cleaned. If the number of triangles is larger than the specified decimate target, the mesh is decimated.

        Parameters:
            path (str): The path to save the mesh.
            density_thresh (int, optional): The density threshold for extracting the mesh. Defaults to 1.
            resolution (int, optional): The resolution of the extracted fields. Defaults to 128.
            decimate_target (float, optional): The target number of triangles for decimation. Defaults to 1e5.

        Returns:
            Mesh: The extracted mesh.
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)

        occ = self.extract_fields(resolution).detach().cpu().numpy()

        import mcubes
        vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1

        # transform back to the original space
        vertices = vertices / self.scale + self.center.detach().cpu().numpy()

        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

        print(
            f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        )

        mesh = Mesh(v=v, f=f, device='cuda')

        return mesh

    def get_covariance(self, scaling_modifier = 1):
        """
        Returns the covariance of the object.

        This method calculates the covariance of the object by applying the scaling modifier to the scaling factor and the rotation.

        Parameters:
            scaling_modifier (int, optional): The scaling modifier for the scaling factor. Defaults to 1.

        Returns:
            Covariance: The covariance of the object.
        """
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """
        Increments the active spherical harmonics degree by one.

        This method increments the active spherical harmonics degree by one if it is less than the maximum spherical harmonics degree.
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float = 1):
        """
        Create a new object from a BasicPointCloud object.

        This function takes a BasicPointCloud object 'pcd' and an optional float parameter 'spatial_lr_scale'.
        The 'spatial_lr_scale' value is assigned to the 'self.spatial_lr_scale' attribute.
        The function then converts the 'points' and 'colors' attributes of the 'pcd' object to tensors,
        performs some calculations, and initializes several attributes of the 'self' object.
        Finally, the function prints the number of points at initialization.

        Parameters:
            pcd (BasicPointCloud): The BasicPointCloud object to create from.
            spatial_lr_scale (float, optional): The spatial lr scale value (default is 1).
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """
        Set up the training for the neural network model.

        This function initializes variables, creates a list of parameters for optimization,
        initializes an optimizer, and sets up a learning rate scheduler.

        Parameters:
            training_args: An object containing training arguments.
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    def update_learning_rate(self, iteration):
        '''
        Learning rate scheduling per step

        Parameters:
            iteration (int): The current iteration.

        Returns:
            float: The updated learning rate.
        '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        '''
        Construct a list of attributes.

        Returns:
            list: A list of attributes.
        '''
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        """
        Save the point cloud in the PLY format.

        This function creates a PLY file at the specified path and saves the point cloud
        data with attributes such as XYZ coordinates, normals, features, opacities, scale, and rotation.

        Parameters:
            path (str): The path where the PLY file will be saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        Reset the opacity values of the point cloud.

        This function updates the opacity values of the point cloud by applying an inverse sigmoid
        to the minimum opacity value of the current point cloud opacity values.
        """
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """
        Load data from a PLY file and assign it to the corresponding class attributes.

        Parameters:
            path (str): The path to the PLY file.
        """
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        Replaces a tensor in the optimizer's state with a new tensor and updates the optimizer's state accordingly.

        Parameters:
            tensor: The new tensor to be inserted into the optimizer's state.
            name (str): The name of the group to which the tensor belongs.

        Returns:
            dict: A dictionary containing the names of the optimizable tensors as keys and the corresponding tensors as values.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        Prune the optimizer based on the given mask.

        This private method iterates over the parameter groups of the optimizer and updates the stored_state,
        optimizer state, and group parameters based on the given mask. It then returns a dictionary of the optimizable tensors.

        Parameters:
            mask (bool): A boolean mask specifying the elements to be pruned.

        Returns:
            dict: A dictionary of the optimizable tensors.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        Apply pruning to the points based on the inverse of the mask.

        This method updates the following attributes of the object:
        - _xyz
        - _features_dc
        - _features_rest
        - _opacity
        - _scaling
        - _rotation
        It also updates the xyz_gradient_accum, denom, and max_radii2D attributes by filtering them based on the valid_points_mask.

        Parameters:
            mask (array-like): A mask indicating which points to prune.
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Concatenate tensors from tensors_dict to the optimizer's state.

        This function takes two parameters: self, a reference to the current object instance, and tensors_dict, a dictionary containing tensors. It iterates over the param_groups of the optimizer and performs various operations on the tensors. It concatenates the extension_tensor with the exp_avg and exp_avg_sq tensors in the optimizer's state dictionary. If the stored_state is not None, it updates the exp_avg and exp_avg_sq tensors and replaces the group's param in the optimizer's state dictionary. Finally, it returns a dictionary of optimizable tensors.

        Parameters:
            self: A reference to the current object instance.
            tensors_dict (dict): A dictionary containing tensors.

        Returns:
            dict: A dictionary of optimizable tensors.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """
        Perform densification postfix operation.

        This method takes in several parameters representing new values for xyz, features_dc, features_rest, opacities, scaling, and rotation. It constructs a dictionary from these values and passes it to the cat_tensors_to_optimizer method. The resulting optimizable_tensors are then assigned to their respective attributes of the object. Finally, the method initializes the xyz_gradient_accum, denom, and max_radii2D attributes with zero tensors.

        Parameters:
            new_xyz: The new xyz values.
            new_features_dc: The new features_dc values.
            new_features_rest: The new features_rest values.
            new_opacities: The new opacities values.
            new_scaling: The new scaling values.
            new_rotation: The new rotation values.
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        Densify and split points based on certain conditions.

        Parameters:
            grads (torch.Tensor): The gradients.
            grad_threshold (float): The gradient threshold.
            scene_extent (float): The extent of the scene.
            N (int, optional): The number of times to repeat the generated data. Defaults to 2.
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        Densify and clone the point cloud based on gradient condition and scaling.

        This function takes in the gradients `grads`, a gradient threshold `grad_threshold`, and the extent of the scene `scene_extent`.
        It extracts the points from `grads` that satisfy the gradient condition and additional conditions based on scaling.
        The extracted points are then used to create new arrays `new_xyz`, `new_features_dc`, `new_features_rest`, `new_opacities`, `new_scaling`, and `new_rotation`.
        The function then calls `densification_postfix` with the newly created arrays as arguments.

        Parameters:
            grads (torch.Tensor): The gradients.
            grad_threshold (float): The gradient threshold.
            scene_extent (float): The extent of the scene.
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        Densifies and prunes the gradient array and then prunes points based on opacity and screen size.

        Parameters:
            max_grad: The maximum gradient value.
            min_opacity: The minimum opacity value.
            extent: The extent of the gradient array.
            max_screen_size: The maximum screen size.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):
        """
        Prunes points based on opacity and screen size.

        Parameters:
            min_opacity: The minimum opacity value.
            extent: The extent of the gradient array.
            max_screen_size: The maximum screen size.
        """

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        Updates the gradient accumulation array and the denominator array based on a gradient tensor.

        Parameters:
            viewspace_point_tensor: The gradient tensor.
            update_filter: The update filter.
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Calculate the projection matrix for a given set of parameters.

    This function calculates the projection matrix for a camera using the specified
    near and far clipping planes, and the horizontal and vertical field of view.

    Parameters:
        znear (float): The distance to the near clipping plane.
        zfar (float): The distance to the far clipping plane.
        fovX (float): The horizontal field of view in radians.
        fovY (float): The vertical field of view in radians.

    Returns:
        torch.Tensor: The projection matrix.
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    """This class represents a camera.

    The camera is initialized with a transformation matrix, width, height, field of view
    values, and near and far clipping planes.

    Attributes:
        image_width (int): The width of the camera image.
        image_height (int): The height of the camera image.
        FoVy (float): The vertical field of view in radians.
        FoVx (float): The horizontal field of view in radians.
        znear (float): The distance to the near clipping plane.
        zfar (float): The distance to the far clipping plane.
        world_view_transform (torch.Tensor): The transformation matrix from world to view coordinates.
        projection_matrix (torch.Tensor): The projection matrix for the camera.
        full_proj_transform (torch.Tensor): The combined world-to-view and projection transform matrix.
        camera_center (torch.Tensor): The center of the camera.
    """
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()


class Renderer:
    """
    This class represents a renderer.
    """
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        """
        Initialize a new Renderer instance.

        Parameters:
            sh_degree (int): The degree of the spherical harmonics.
            white_background (bool): Whether to use a white background.
            radius (int): The radius.
        """

        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

    def initialize(self, input=None, num_pts=5000, radius=0.5):
        """
        Initialize the point cloud.

        This function initializes a point cloud with random points and colors.

        Parameters:
            input (BasicPointCloud, optional): An optional input point cloud.
            num_pts (int, optional): The number of points in the cloud. Default is 5000.
            radius (float, optional): The radius of the points. Default is 0.5.
        """
        # load checkpoint
        if input is None:
            # init from random point cloud

            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 10)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1)
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        """
        Render the image of Gaussians.

        This function takes in various parameters and performs a series of calculations and operations to render an image of Gaussians. It initializes a tensor for screen-space points and sets up the rasterization configuration. It then processes the means, opacity, scales, rotations, covariance, and colors of the Gaussians. Finally, it rasterizes the visible Gaussians, clamps the rendered image, and returns the rendered image, depth, alpha, screen-space points, visibility filter, and radii.

        Parameters:
            viewpoint_camera: The viewpoint camera.
            scaling_modifier (float): The scaling modifier.
            invert_bg_color (bool): Whether to invert the background color.
            override_color: The overridden color.
            compute_cov3D_python (bool): Whether to compute the 3D covariance in Python.
            convert_SHs_python (bool): Whether to convert SHs in Python.

        Returns:
            dict: A dictionary containing the rendered image, depth, alpha, screen-space points, visibility filter, and radii.
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if not invert_bg_color else 1 - self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

