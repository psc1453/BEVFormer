# Projects.mmdet_3d_plugin.bevformer.modules.encoder.BEVFormerEncoder.get_reference_points

import torch

H = 2
W = 5
Z = 8
bs = 1

num_points_in_pillar = 4


def get_3d_ref():
    zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=torch.float)
    print(zs)
    zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=torch.float).view(-1, 1, 1)
    print(zs)
    # Values are same on same Z plane, step along Z (dimension 0, towards sky) by line space
    zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=torch.float).view(-1, 1, 1).expand(
        num_points_in_pillar, H, W) / Z
    print(zs)

    xs = torch.linspace(0.5, W - 0.5, W, dtype=torch.float)
    print(xs)
    xs = torch.linspace(0.5, W - 0.5, W, dtype=torch.float).view(1, 1, W)
    print(xs)
    # Values are same on same X plane, step along x (dimension 2, towards side)
    xs = torch.linspace(0.5, W - 0.5, W, dtype=torch.float).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
    print(xs)

    ys = torch.linspace(0.5, H - 0.5, H, dtype=torch.float)
    print(ys)
    ys = torch.linspace(0.5, H - 0.5, H, dtype=torch.float).view(1, H, 1)
    print(ys)
    # Values are same on same Y plane, step along x (dimension 1, towards front) by line space
    ys = torch.linspace(0.5, H - 0.5, H, dtype=torch.float).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
    print(ys)

    # The following 2 steps are equivalent to: torch.stack((xs, ys, zs),1)
    # Imagine that the whole space is Z x W x H
    # There are 3 cube withe gradients along different axis
    # Fuse them in the last dimension so that they act as an element
    # Growing direction of xs, ys, zs only decides access axis of them, stack order decides order of vector-element
    # num_points_in_pillar x H x W x 3
    ref_3d = torch.stack((xs, ys, zs), -1)
    print(ref_3d)
    # Put H & W dimensions to the tail for later flatten
    # num_points_in_pillar x 3 x H x W
    ref_3d = ref_3d.permute(0, 3, 1, 2)
    print(ref_3d.shape)

    # Stretch the H & W plane to 1D vector
    # num_points_in_pillar x 3 x (H x W)
    ref_3d = ref_3d.flatten(2)
    print(ref_3d.shape)
    # num_points_in_pillar x (H x W) x 3
    ref_3d = ref_3d.permute(0, 2, 1)

    # Broadcast value to create a first dimension
    # Equivalent to ref_3d.reshape([1] + ref_3d.shape)
    # 1 x num_points_in_pillar x (H x W) x 3
    ref_3d = ref_3d[None]
    print(ref_3d.shape)

    ref_3d = ref_3d.repeat(bs, 1, 1, 1)
    print(ref_3d.shape)
    return ref_3d


def get_2d_ref():
    ys = torch.linspace(0.5, H - 0.5, H, dtype=torch.float) / H
    xs = torch.linspace(0.5, W - 0.5, W, dtype=torch.float) / W
    print(ys)
    print(xs)

    # H x W both
    # Elements in xs and ys are y and x coordinate in mesh
    ref_y, ref_x = torch.meshgrid(ys, xs, indexing='ij')
    print(ref_y)
    print(ref_x)

    # Flatten HW dimension and add a dimension to the first place
    # The following 3 steps are equivalent to torch.stack((ref_x, ref_y), -1).reshape(1, -1, 2)
    # 1 x (H x W)
    ref_y = ref_y.reshape(-1)[None]
    # 1 x (H x W)
    ref_x = ref_x.reshape(-1)[None]
    # 1 x (H x W) x 2
    ref_2d = torch.stack((ref_x, ref_y), -1)

    # Add a dimension to dimension 2
    # 1 x (H x W) x 1 x 2
    ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    print(ref_2d.shape)

    return ref_2d
