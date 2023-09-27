import torch
import torch.nn.functional as F

def stride_from_shape(shape):
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x) 
    return list(reversed(stride))


def scatter_add_nd(input, indices, values):
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