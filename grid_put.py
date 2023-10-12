import torch
import torch.nn.functional as F

def stride_from_shape(shape):
    """
    Calculate the stride for each dimension in reverse order.

    Args:
        shape (List[int]): The shape of the tensor.

    Returns:
        List[int]: The strides for each dimension in reverse order.
    """
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x) 
    return list(reversed(stride))


def scatter_add_nd(input, indices, values):
    """
    Perform scatter addition on the input tensor using indices and values.

    Args:
        input (torch.Tensor): The input tensor of shape [..., C].
        indices (torch.Tensor): The indices tensor of shape [N, D].
        values (torch.Tensor): The values tensor of shape [N, C].

    Returns:
        torch.Tensor: The updated input tensor after scatter addition.
    """
    # input: [..., C], D dimension + C channel
    # indices: [N, D], long
    # values: [N, C]

    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)

    assert len(size) == D

    input = input.view(-1, C)  # [HW, C]
    flatten_indices = (indices * torch.tensor(stride, dtype=torch.long, device=indices.device)).sum(-1)  # [N]

    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)

    return input.view(*size, C)


def scatter_add_nd_with_count(input, count, indices, values, weights=None):
    """
    Perform a scatter add operation on a multi-dimensional input tensor.

    This function takes several input arguments including 'input', 'count', 'indices', 'values', and 'weights' (optional).
    It reshapes the input and count tensors, computes the flatten indices, and performs scatter add operations on the input and count tensors using the flatten indices.
    Finally, it returns the reshaped input and count tensors.

    Parameters:
        input (Tensor): The input tensor of shape [..., C], where D is the dimension and C is the channel.
        count (Tensor): The count tensor of shape [..., 1], where D is the dimension.
        indices (Tensor): The indices tensor of shape [N, D], where N is the number of indices and D is the dimension.
        values (Tensor): The values tensor of shape [N, C], where C is the channel.
        weights (Tensor, optional): The weights tensor of shape [N, 1]. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the reshaped input tensor of shape [..., C] and the reshaped count tensor of shape [..., 1].
    """
    # input: [..., C], D dimension + C channel
    # count: [..., 1], D dimension
    # indices: [N, D], long
    # values: [N, C]

    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)

    assert len(size) == D

    input = input.view(-1, C)  # [HW, C]
    count = count.view(-1, 1)

    flatten_indices = (indices * torch.tensor(stride, dtype=torch.long, device=indices.device)).sum(-1)  # [N]

    if weights is None:
        weights = torch.ones_like(values[..., :1]) 

    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)
    count.scatter_add_(0, flatten_indices.unsqueeze(1), weights)

    return input.view(*size, C), count.view(*size, 1)

def nearest_grid_put_2d(H, W, coords, values, return_count=False):
    """
    Perform nearest grid interpolation by mapping the input coordinates to
    the nearest grid indices and scatter-adding the input values to the
    result tensor at the corresponding grid indices.

    Parameters:
        H (int): The height of the result tensor.
        W (int): The width of the result tensor.
        coords (torch.Tensor): The input coordinates, a 2D tensor of shape (N, 2) where N is the number of coordinates and each coordinate is a float in the range [-1, 1].
        values (torch.Tensor): The input values, a 2D tensor of shape (N, C) where C is the number of values.
        return_count (bool, optional): Whether to return the count tensor. Defaults to False.

    Returns:
        torch.Tensor: The result tensor of shape (H, W, C).
        torch.Tensor, optional: The count tensor of shape (H, W, 1), returned only if return_count is True.
    """
    # coords: [N, 2], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = (coords * 0.5 + 0.5) * torch.tensor(
        [H - 1, W - 1], dtype=torch.float32, device=coords.device
    )
    indices = indices.round().long()  # [N, 2]

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]
    weights = torch.ones_like(values[..., :1])  # [N, 1]

    result, count = scatter_add_nd_with_count(result, count, indices, values, weights)

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


def linear_grid_put_2d(H, W, coords, values, return_count=False):
    """
    Perform linear grid interpolation on a 2D grid.

    This function takes as input the dimensions of the grid (H and W), the coordinates of the points to interpolate (coords), and the values associated with each point (values). It returns the interpolated grid. If the return_count flag is set to True, it also returns the count of values used for each grid cell.

    Parameters:
        H (int): The height of the grid.
        W (int): The width of the grid.
        coords (torch.Tensor): The coordinates of the points to interpolate, with values in the range [-1, 1].
        values (torch.Tensor): The values associated with each point.
        return_count (bool, optional): Whether to return the count of values used for each grid cell.

    Returns:
        torch.Tensor: The interpolated grid.
        torch.Tensor: The count of values used for each grid cell, if return_count is True.
    """
    # coords: [N, 2], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = (coords * 0.5 + 0.5) * torch.tensor(
        [H - 1, W - 1], dtype=torch.float32, device=coords.device
    )
    indices_00 = indices.floor().long()  # [N, 2]
    indices_00[:, 0].clamp_(0, H - 2)
    indices_00[:, 1].clamp_(0, W - 2)
    indices_01 = indices_00 + torch.tensor(
        [0, 1], dtype=torch.long, device=indices.device
    )
    indices_10 = indices_00 + torch.tensor(
        [1, 0], dtype=torch.long, device=indices.device
    )
    indices_11 = indices_00 + torch.tensor(
        [1, 1], dtype=torch.long, device=indices.device
    )

    h = indices[..., 0] - indices_00[..., 0].float()
    w = indices[..., 1] - indices_00[..., 1].float()
    w_00 = (1 - h) * (1 - w)
    w_01 = (1 - h) * w
    w_10 = h * (1 - w)
    w_11 = h * w

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]
    weights = torch.ones_like(values[..., :1])  # [N, 1]

    result, count = scatter_add_nd_with_count(result, count, indices_00, values * w_00.unsqueeze(1), weights* w_00.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_01, values * w_01.unsqueeze(1), weights* w_01.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_10, values * w_10.unsqueeze(1), weights* w_10.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_11, values * w_11.unsqueeze(1), weights* w_11.unsqueeze(1))

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result

def mipmap_linear_grid_put_2d(H, W, coords, values, min_resolution=32, return_count=False):
    """
    Perform a mipmap operation on a 2D grid of coordinates and values.

    This function iteratively performs a downsampling operation until the resolution of the grid
    reaches the minimum resolution specified by min_resolution. During each iteration, the function
    fills in the empty regions of the grid by interpolating the values from a smaller grid. Finally,
    the function normalizes the values by dividing them by the count of non-zero elements.

    Parameters:
        H (int): The height of the grid.
        W (int): The width of the grid.
        coords (torch.Tensor): The coordinates of the grid, in the range [-1, 1]. Shape: [N, 2]
        values (torch.Tensor): The values associated with the coordinates. Shape: [N, C]
        min_resolution (int, optional): The minimum resolution of the grid. Defaults to 32.
        return_count (bool, optional): Whether to return the count of non-zero elements. Defaults to False.

    Returns:
        torch.Tensor: The resulting grid after the mipmap operation. Shape: [H, W, C]
        torch.Tensor: The count of non-zero elements, if return_count is True. Shape: [H, W, 1]
    """
    # coords: [N, 2], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]

    cur_H, cur_W = H, W

    while min(cur_H, cur_W) > min_resolution:

        # try to fill the holes
        mask = (count.squeeze(-1) == 0)
        if not mask.any():
            break

        cur_result, cur_count = linear_grid_put_2d(cur_H, cur_W, coords, values, return_count=True)
        result[mask] = result[mask] + F.interpolate(cur_result.permute(2,0,1).unsqueeze(0).contiguous(), (H, W), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0).contiguous()[mask]
        count[mask] = count[mask] + F.interpolate(cur_count.view(1, 1, cur_H, cur_W), (H, W), mode='bilinear', align_corners=False).view(H, W, 1)[mask]
        cur_H //= 2
        cur_W //= 2

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result

def nearest_grid_put_3d(H, W, D, coords, values, return_count=False):
    """Nearest neighbor interpolation for 3D grid data.

    This function takes in the dimensions of the grid (H, W, D), the coordinates of the data points (coords), the values associated with the data points (values), and an optional flag to return the count of interpolated points (return_count).

    Parameters:
        H (int): The height of the grid.
        W (int): The width of the grid.
        D (int): The depth of the grid.
        coords (torch.Tensor): The coordinates of the data points, as a tensor of shape [N, 3]. The coordinates should be in the range [-1, 1].
        values (torch.Tensor): The values associated with the data points, as a tensor of shape [N, C].
        return_count (bool, optional): Flag to return the count of interpolated points. Defaults to False.

    Returns:
        torch.Tensor: The interpolated grid data, as a tensor of shape [H, W, D, C].
        torch.Tensor: The count of interpolated points, as a tensor of shape [H, W, D, 1].
    """
    # coords: [N, 3], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = (coords * 0.5 + 0.5) * torch.tensor(
        [H - 1, W - 1, D - 1], dtype=torch.float32, device=coords.device
    )
    indices = indices.round().long()  # [N, 2]

    result = torch.zeros(H, W, D, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, D, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]
    weights = torch.ones_like(values[..., :1])  # [N, 1]

    result, count = scatter_add_nd_with_count(result, count, indices, values, weights)

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


def linear_grid_put_3d(H, W, D, coords, values, return_count=False):
    """
    Perform a 3D scatter operation on a grid.

    This function takes as input the height (H), width (W), and depth (D) of the grid,
    as well as the coordinates (coords) and values (values) of the data points to scatter.
    It calculates the indices of the grid cells that each coordinate falls into, and
    computes the weights for each cell based on the distance between the coordinate
    and the indices. Finally, it uses the indices, weights, and values to update the
    result grid and counts the number of data points that fall into each cell.

    Parameters:
        H (int): The height of the grid.
        W (int): The width of the grid.
        D (int): The depth of the grid.
        coords (torch.Tensor): The coordinates of the data points to scatter.
        values (torch.Tensor): The values of the data points to scatter.
        return_count (bool, optional): Whether to return the count grid.
            Defaults to False.

    Returns:
        result (torch.Tensor): The updated result grid.
        count (torch.Tensor, optional): The count grid if return_count is True.
    """
    # coords: [N, 3], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = (coords * 0.5 + 0.5) * torch.tensor(
        [H - 1, W - 1, D - 1], dtype=torch.float32, device=coords.device
    )
    indices_000 = indices.floor().long()  # [N, 3]
    indices_000[:, 0].clamp_(0, H - 2)
    indices_000[:, 1].clamp_(0, W - 2)
    indices_000[:, 2].clamp_(0, D - 2)

    indices_001 = indices_000 + torch.tensor([0, 0, 1], dtype=torch.long, device=indices.device)
    indices_010 = indices_000 + torch.tensor([0, 1, 0], dtype=torch.long, device=indices.device)
    indices_011 = indices_000 + torch.tensor([0, 1, 1], dtype=torch.long, device=indices.device)
    indices_100 = indices_000 + torch.tensor([1, 0, 0], dtype=torch.long, device=indices.device)
    indices_101 = indices_000 + torch.tensor([1, 0, 1], dtype=torch.long, device=indices.device)
    indices_110 = indices_000 + torch.tensor([1, 1, 0], dtype=torch.long, device=indices.device)
    indices_111 = indices_000 + torch.tensor([1, 1, 1], dtype=torch.long, device=indices.device)

    h = indices[..., 0] - indices_000[..., 0].float()
    w = indices[..., 1] - indices_000[..., 1].float()
    d = indices[..., 2] - indices_000[..., 2].float()

    w_000 = (1 - h) * (1 - w) * (1 - d)
    w_001 = (1 - h) * w * (1 - d)
    w_010 = h * (1 - w) * (1 - d)
    w_011 = h * w * (1 - d)
    w_100 = (1 - h) * (1 - w) * d
    w_101 = (1 - h) * w * d
    w_110 = h * (1 - w) * d
    w_111 = h * w * d

    result = torch.zeros(H, W, D, C, device=values.device, dtype=values.dtype)  # [H, W, D, C]
    count = torch.zeros(H, W, D, 1, device=values.device, dtype=values.dtype)  # [H, W, D, 1]
    weights = torch.ones_like(values[..., :1])  # [N, 1]

    result, count = scatter_add_nd_with_count(result, count, indices_000, values * w_000.unsqueeze(1), weights * w_000.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_001, values * w_001.unsqueeze(1), weights * w_001.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_010, values * w_010.unsqueeze(1), weights * w_010.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_011, values * w_011.unsqueeze(1), weights * w_011.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_100, values * w_100.unsqueeze(1), weights * w_100.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_101, values * w_101.unsqueeze(1), weights * w_101.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_110, values * w_110.unsqueeze(1), weights * w_110.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_111, values * w_111.unsqueeze(1), weights * w_111.unsqueeze(1))

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result

def mipmap_linear_grid_put_3d(H, W, D, coords, values, min_resolution=32, return_count=False):
    """
    Perform a mipmap operation on a 3D grid.

    This function takes the following input parameters:
    - H: An integer representing the height of the grid.
    - W: An integer representing the width of the grid.
    - D: An integer representing the depth of the grid.
    - coords: A 2D array of shape (N, 3) containing float values in the range [-1, 1], where N is the number of coordinates.
    - values: A 2D array of shape (N, C) containing values associated with each coordinate, where C is the number of channels.
    - min_resolution: An optional integer representing the minimum resolution of the grid.
    - return_count: An optional boolean indicating whether to return the count of filled cells.

    The function performs a downsampling operation on the input grid using a linear interpolation scheme. It starts with the original grid size and iteratively reduces the size by half until the minimum resolution is reached.

    It uses the 'linear_grid_put_3d' function to perform the downsampling for each iteration. The 'linear_grid_put_3d' function takes the current grid size, coordinates, and values as input and returns the interpolated result as well as the count of filled cells.

    The function fills the holes in the result grid by applying the downsampling operation only to the empty cells. It updates the 'result' and 'count' tensors accordingly.

    Finally, if 'return_count' is True, the function returns both the 'result' and 'count' tensors. Otherwise, it normalizes the 'result' tensor by dividing it by the count of filled cells and returns the result.
    """
    # coords: [N, 3], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    result = torch.zeros(H, W, D, C, device=values.device, dtype=values.dtype)  # [H, W, D, C]
    count = torch.zeros(H, W, D, 1, device=values.device, dtype=values.dtype)  # [H, W, D, 1]
    cur_H, cur_W, cur_D = H, W, D

    while min(min(cur_H, cur_W), cur_D) > min_resolution:

        # try to fill the holes
        mask = (count.squeeze(-1) == 0)
        if not mask.any():
            break

        cur_result, cur_count = linear_grid_put_3d(cur_H, cur_W, cur_D, coords, values, return_count=True)
        result[mask] = result[mask] + F.interpolate(cur_result.permute(3,0,1,2).unsqueeze(0).contiguous(), (H, W, D), mode='trilinear', align_corners=False).squeeze(0).permute(1,2,3,0).contiguous()[mask]
        count[mask] = count[mask] + F.interpolate(cur_count.view(1, 1, cur_H, cur_W, cur_D), (H, W, D), mode='trilinear', align_corners=False).view(H, W, D, 1)[mask]
        cur_H //= 2
        cur_W //= 2
        cur_D //= 2

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


def grid_put(shape, coords, values, mode='linear-mipmap', min_resolution=32, return_raw=False):
    """
    Perform grid put operation based on the given parameters.

    This function takes in a shape, coordinates, values, mode, minimum resolution, and return_raw flag. It performs the grid put operation based on the input parameters and returns the result.

    Parameters:
        shape (list/tuple): The shape of the grid.
        coords (numpy.ndarray): The coordinates of the grid points.
        values (numpy.ndarray): The values to be placed in the grid.
        mode (str, optional): The mode of the grid put operation. Defaults to 'linear-mipmap'.
        min_resolution (int, optional): The minimum resolution of the grid. Defaults to 32.
        return_raw (bool, optional): Flag indicating whether to return raw result. Defaults to False.

    Returns:
        numpy.ndarray: The result of the grid put operation.
    """
    # shape: [D], list/tuple
    # coords: [N, D], float in [-1, 1]
    # values: [N, C]

    D = len(shape)
    assert D in [2, 3], f'only support D == 2 or 3, but got D == {D}'

    if mode == 'nearest':
        if D == 2:
            return nearest_grid_put_2d(*shape, coords, values, return_raw)
        else:
            return nearest_grid_put_3d(*shape, coords, values, return_raw)
    elif mode == 'linear':
        if D == 2:
            return linear_grid_put_2d(*shape, coords, values, return_raw)
        else:
            return linear_grid_put_3d(*shape, coords, values, return_raw)
    elif mode == 'linear-mipmap':
        if D == 2:
            return mipmap_linear_grid_put_2d(*shape, coords, values, min_resolution, return_raw)
        else:
            return mipmap_linear_grid_put_3d(*shape, coords, values, min_resolution, return_raw)
    else:
        raise NotImplementedError(f"got mode {mode}")    