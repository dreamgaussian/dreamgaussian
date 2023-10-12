import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

def dot(x, y):
    """
    Compute the dot product between two vectors.

    Parameters:
        x (ndarray or Tensor): The first vector.
        y (ndarray or Tensor): The second vector.

    Returns:
        ndarray or Tensor: The dot product of x and y.
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    """
    Calculate the length (magnitude) of a vector.

    Parameters:
        x (ndarray or Tensor): The vector.
        eps (float, optional): A small epsilon value to avoid division by zero.

    Returns:
        ndarray or Tensor: The length of x.
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    """
    Safely normalize a vector to have unit length.

    Parameters:
        x (ndarray or Tensor): The vector to be normalized.
        eps (float, optional): A small epsilon value to avoid division by zero.

    Returns:
        ndarray or Tensor: The normalized vector.
    """
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    """
    Compute the rotation matrix that aligns the camera with the target object.

    Parameters:
        campos (ndarray): The camera/eye position.
        target (ndarray): The object to look at.
        opengl (bool, optional): Whether to use OpenGL convention or not.

    Returns:
        ndarray: The rotation matrix.
    """
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    """
    Calculate the camera pose matrix from elevation, azimuth, and radius.

    This function calculates the camera pose matrix based on the given elevation, azimuth, and radius values. The camera pose matrix represents the transformation from camera coordinates to world coordinates.

    Parameters:
        elevation (float): The elevation angle in degrees or radians, depending on the value of 'is_degree'. Should be in the range (-90, 90), where +y corresponds to -90 degrees and -y corresponds to +90 degrees.
        azimuth (float): The azimuth angle in degrees or radians, depending on the value of 'is_degree'. Should be in the range (-180, 180), where +z corresponds to 0 degrees and +x corresponds to 90 degrees.
        radius (float, optional): The radius of the camera orbit. Defaults to 1.
        is_degree (bool, optional): Indicates whether the input angles are in degrees. If True, the input angles are converted to radians. Defaults to True.
        target (numpy.ndarray, optional): The target point in world coordinates. If None, the target point is set to the origin. Defaults to None.
        opengl (bool, optional): Indicates whether the camera pose matrix should be computed for OpenGL convention. If True, the camera pose matrix is computed for OpenGL convention. Defaults to True.

    Returns:
        numpy.ndarray: The camera pose matrix as a 4x4 numpy array.
    """
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


class OrbitCamera:
    """
    Class representing a camera in a 3D scene.
    """
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        """
        Initialize the OrbitCamera with specified attributes.

        Parameters:
            W (int): Width of the camera viewport.
            H (int): Height of the camera viewport.
            r (float, optional): Camera distance from center. Defaults to 2.
            fovy (float, optional): Field of view in the y-direction. Defaults to 60.
            near (float, optional): Near clipping plane distance. Defaults to 0.01.
            far (float, optional): Far clipping plane distance. Defaults to 100.
        """
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        """
        Calculate the field of view in the x-direction.

        Returns:
            float: Field of view in the x-direction.
        """
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        """
        Get the camera position.

        Returns:
            ndarray: Camera position.
        """
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        """
        Get the camera pose (c2w).

        Returns:
            ndarray: Camera pose.
        """
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        """
        Get the camera view matrix (w2c).

        Returns:
            ndarray: Camera view matrix.
        """
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        """
        Return the perspective projection matrix.

        This method calculates and returns a 4x4 perspective projection matrix based
        on the field of view, aspect ratio, and near/far clipping planes.

        Returns:
            np.ndarray: A 4x4 perspective projection matrix.
        """
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        """
        Return the camera intrinsics.

        This method calculates and returns the camera intrinsics as a 1D array.
        The intrinsics consist of the focal length, principal point coordinates,
        and image dimensions.

        Returns:
            np.ndarray: The camera intrinsics as a 1D array.
        """
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        """
        Return the model-view-projection matrix.

        This method calculates and returns the model-view-projection (MVP) matrix
        by combining the perspective projection matrix with the inverse of the camera pose.

        Returns:
            np.ndarray: The model-view-projection matrix.
        """
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        """
        Rotate the camera around its up and side axes.

        This method rotates the camera by the given delta values along the camera's
        up and side axes.

        Parameters:
            dx (float): The rotation angle along the camera's up axis.
            dy (float): The rotation angle along the camera's side axis.
        """
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        """
        Adjust the camera radius.

        This method adjusts the camera radius by the given delta value.

        Parameters:
            delta (float): The change in camera radius.
        """
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        """
        Pan the camera in the camera coordinate system.

        This method pans the camera in the camera coordinate system by the given
        delta values.

        Parameters:
            dx (float): The pan distance along the camera's x-axis.
            dy (float): The pan distance along the camera's y-axis.
            dz (float, optional): The pan distance along the camera's z-axis.
                Defaults to 0.
        """
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])