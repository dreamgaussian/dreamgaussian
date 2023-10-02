import os
import cv2
import numpy as np
import trimesh

import torch
import torch.nn.functional as F

def dot(x, y):
    """
    Calculate the dot product of two torch tensors.

    Parameters:
        x (torch.Tensor): The first tensor.
        y (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The dot product of x and y.
    """
    return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    """
    Calculate the length of a torch tensor.

    Parameters:
        x (torch.Tensor): The tensor.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The length of x.
    """
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    """
    Normalize a torch tensor.

    Parameters:
        x (torch.Tensor): The tensor.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    return x / length(x, eps)


class Mesh:
    """This class represents a mesh object and provides methods for loading and manipulating mesh data."""
    def __init__(
        self,
        v=None,
        f=None,
        vn=None,
        fn=None,
        vt=None,
        ft=None,
        albedo=None,
        device=None,
    ):
        """Initialize a new Mesh object.

        Parameters:
            v: Vertex data of the mesh.
            f: Face data of the mesh.
            vn: Normal data of the mesh.
            fn: Face normal data of the mesh.
            vt: Texture coordinate data of the mesh.
            ft: Face texture coordinate data of the mesh.
            albedo: Albedo data of the mesh.
            device: Device used for the mesh.
        """
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        # only support a single albedo
        self.albedo = albedo

        self.ori_center = 0
        self.ori_scale = 1

    @classmethod
    def load(cls, path=None, resize=True, **kwargs):
        """Load mesh data from a file.

        This method assumes initialization with keyword arguments.

        Parameters:
            path: Path to the file containing mesh data.
            resize: Flag indicating whether to perform automatic resizing.
            **kwargs: Additional keyword arguments for initializing the Mesh object.

        Returns:
            Mesh: The loaded Mesh object.
        """
        # assume init with kwargs
        if path is None:
            mesh = cls(**kwargs)
        # obj supports face uv
        elif path.endswith(".obj"):
            mesh = cls.load_obj(path, **kwargs)
        # trimesh only supports vertex uv, but can load more formats
        else:
            mesh = cls.load_trimesh(path, **kwargs)

        print(f"[Mesh loading] v: {mesh.v.shape}, f: {mesh.f.shape}")
        # auto-normalize
        if resize:
            mesh.auto_size()
        # auto-fix normal
        if mesh.vn is None:
            mesh.auto_normal()
        print(f"[Mesh loading] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")
        # auto-fix texture
        if mesh.vt is None:
            mesh.auto_uv(cache_path=path)
        print(f"[Mesh loading] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")

        return mesh

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None, init_empty_tex=False):
        """Load a 3D mesh object from an OBJ file.

        This method loads a 3D mesh object from the specified OBJ file path. The method takes several parameters including the file path, an optional albedo texture path, the device to use (default is 'cuda' if available, otherwise 'cpu'), and a flag to initialize an empty texture if no albedo path is provided. The method reads the OBJ file and extracts the vertex positions, texture coordinates, and normals. It also attempts to find the albedo texture path from the corresponding MTL file or falls back to an empty texture if no albedo path is found or provided. The method then loads the albedo texture, converts it to the RGB color space, and stores it in the mesh object. Finally, the method returns the mesh object.

        Parameters:
            path (str): The file path of the OBJ file.
            albedo_path (str, optional): The file path of the albedo texture. Defaults to None.
            device (torch.device, optional): The device to use. Defaults to None.
            init_empty_tex (bool, optional): Whether to initialize an empty texture. Defaults to False.

        Returns:
            mesh: The loaded mesh object.
        """
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # try to find texture from mtl file
        if albedo_path is None:
            mtl_path = path.replace(".obj", ".mtl")
            if os.path.exists(mtl_path):
                with open(mtl_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0:
                        continue
                    prefix = split_line[0]
                    # NOTE: simply use the first map_Kd as albedo!
                    if "map_Kd" in prefix:
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        print(f"[load_obj] use texture from: {albedo_path}")
                        break

        if init_empty_tex or albedo_path is None or not os.path.exists(albedo_path):
            # init an empty texture
            print(f"[load_obj] init empty albedo!")
            # albedo = np.random.rand(1024, 1024, 3).astype(np.float32)
            albedo = np.ones((1024, 1024, 3), dtype=np.float32) * np.array(
                [0.5, 0.5, 0.5]
            )  # default color
        else:
            albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
            albedo = albedo.astype(np.float32) / 255
            print(f"[load_obj] load texture: {albedo.shape}")

            # import matplotlib.pyplot as plt
            # plt.imshow(albedo)
            # plt.show()

        mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)

        # load obj
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            # v/vn/vt
            prefix = split_line[0].lower()
            if prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == "vt":
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == "f":
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh

    @classmethod
    def load_trimesh(cls, path, device=None):
        """
        Load a Trimesh object from a specified path.

        This method loads a Trimesh object from the specified path. It checks the type of the loaded data and extracts the mesh and material information. It then converts the material texture into a torch tensor and assigns it to the 'albedo' attribute of the mesh. The vertices, texture coordinates, normals, and faces of the mesh are extracted and assigned to the appropriate attributes of the mesh object.

        Parameters:
            path (str): The path to the mesh file.
            device: The device to be used for tensor operations. Defaults to None.

        Returns:
            Trimesh: The loaded Trimesh object.
        """
        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # use trimesh to load glb, assume only has one single RootMesh...
        _data = trimesh.load(path)
        if isinstance(_data, trimesh.Scene):
            mesh_keys = list(_data.geometry.keys())
            assert (
                len(mesh_keys) == 1
            ), f"{path} contains more than one meshes, not supported!"
            _mesh = _data.geometry[mesh_keys[0]]

        elif isinstance(_data, trimesh.Trimesh):
            _mesh = _data

        else:
            raise NotImplementedError(f"type {type(_data)} not supported!")

        # TODO: exception handling if no material
        _material = _mesh.visual.material
        if isinstance(_material, trimesh.visual.material.PBRMaterial):
            texture = np.array(_material.baseColorTexture).astype(np.float32) / 255
        elif isinstance(_material, trimesh.visual.material.SimpleMaterial):
            texture = (
                np.array(_material.to_pbr().baseColorTexture).astype(np.float32) / 255
            )
        else:
            raise NotImplementedError(f"material type {type(_material)} not supported!")

        print(f"[load_obj] load texture: {texture.shape}")
        mesh.albedo = torch.tensor(texture, dtype=torch.float32, device=device)

        vertices = _mesh.vertices
        texcoords = _mesh.visual.uv
        texcoords[:, 1] = 1 - texcoords[:, 1]
        normals = _mesh.vertex_normals

        # trimesh only support vertex uv...
        faces = tfaces = nfaces = _mesh.faces

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh

    # aabb
    def aabb(self):
        """Calculate the axis-aligned bounding box of the data.

        Returns:
            (torch.Tensor, torch.Tensor): A tuple of two tensors representing the minimum and maximum values along each dimension.
        """
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self):
        """Rescale the data to fit within a specified range.

        The rescaling is performed by calculating the minimum and maximum values of the data, and then transforming the data so that the minimum value maps to -0.6 and the maximum value maps to 0.6.
        """
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 1.2 / torch.max(vmax - vmin).item() # to ~ [-0.6, 0.6]
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal(self):
        """Calculate the vertex normals of a mesh.

        The vertex normals are calculated by first calculating the face normals of each triangle in the mesh, and then averaging the face normals for each vertex. The resulting vertex normals are then normalized.
        """
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        self.vn = vn
        self.fn = self.f

    def auto_uv(self, cache_path=None):
        """
        Generate UV coordinates for a 3D mesh.

        This function tries to load the UV coordinates from a cache file specified by 'cache_path'.
        If the cache file does not exist, it generates the UV coordinates using the 'xatlas' library.
        The UV coordinates are then saved to the cache file for future use.

        Parameters:
            cache_path (str, optional): The path to the cache file. Defaults to None.

        Returns:
            None
        """
        # try to load cache
        if cache_path is not None:
            cache_path = cache_path.replace(".obj", "_uv.npz")

        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np = data["vt"], data["ft"]
        else:
            import xatlas

            v_np = self.v.detach().cpu().numpy()
            f_np = self.f.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            # chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path is not None:
                np.savez(cache_path, vt=vt_np, ft=ft_np)

        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)

        self.vt = vt
        self.ft = ft

    def to(self, device):
        """
        Transfer the mesh data to the specified device.

        This method is used to transfer the vertex, face, normal, texture, and albedo data to a specified device.

        Parameters:
            device: The device to transfer the data to.

        Returns:
            self
        """
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self

    # write to ply file (only geom)
    def write_ply(self, path):
        """
        Write the mesh to a PLY file.

        Parameters:
            path (str): The file path to write to. Must have the '.ply' extension.
        """
        assert path.endswith(".ply")

        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        _mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        _mesh.export(path)

    # write to obj file
    def write(self, path):
        """Write data to OBJ and MTL files.

        This function writes the vertex, texture coordinate, normal, and face data to an OBJ file, and
        writes material properties and the albedo image path to an MTL file. The function also saves
        the albedo image as a PNG file.

        Parameters:
            path (str): The path to the OBJ file.
        """
        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = path.replace(".obj", "_albedo.png")

        v_np = self.v.detach().cpu().numpy()
        vt_np = self.vt.detach().cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.detach().cpu().numpy() if self.vn is not None else None
        f_np = self.f.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.detach().cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f"mtllib {os.path.basename(mtl_path)} \n")

            for v in v_np:
                fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

            if vt_np is not None:
                for v in vt_np:
                    fp.write(f"vt {v[0]} {1 - v[1]} \n")

            if vn_np is not None:
                for v in vn_np:
                    fp.write(f"vn {v[0]} {v[1]} {v[2]} \n")

            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n'
                )

        with open(mtl_path, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")
            fp.write(f"map_Kd {os.path.basename(albedo_path)} \n")

        albedo = self.albedo.detach().cpu().numpy()
        albedo = (albedo * 255).astype(np.uint8)
        cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
