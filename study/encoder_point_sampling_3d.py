# Projects.mmdet_3d_plugin.bevformer.modules.encoder.BEVFormerEncoder.point_sampling

import torch

from encoder_get_reference_points import get_3d_ref
from utils import no_output


def point_sampling_3d():
    # [x_start, y_start, z_start, x_end, y_end, z_end]
    pc_range = [0, 0, 0, 2, 3, 4]
    # The rotate-translation matrix (fake shape)
    lidar2img = torch.randn([1, 8, 16])

    reference_points: torch.Tensor = no_output(get_3d_ref)()
    # Clone it to prevent modification passed to the source
    # bs x num_points_in_pillar x (H x W) x 3 -> 1 x 4 x 10 x 3
    reference_points = reference_points.clone()
    print(reference_points)

    # The last dimension is like vector-element, transform the range of each element in vector-element
    # When only 2 elements in vector-element, reference_points[2] cause error while reference_points[2:3] get empty tensor
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    print(reference_points)

    # Generate a last-value for vector-element with the same former dimension
    ones_tensor = torch.ones_like(reference_points[..., :1])
    # Concat the ones tensor to original tensor so that each vector-elements are appended by a new value: 1
    # 1 x 4 x 10 x 4
    reference_points = torch.cat((reference_points, ones_tensor), -1)
    print(reference_points.shape)

    # Exchange dimension 0 and 1
    # num_points_in_pillar x bs x (H x W) x 3 -> 4 x 1 x 10 x 3
    reference_points = reference_points.permute(1, 0, 2, 3)
    print(reference_points.shape)

    # Get D: number of sample points, B: batch size. num_query: number of bev points
    D, B, num_query = reference_points.size()[:3]
    print(f'D: {D}, B:{B}, num_query:{num_query}')
    num_cam = lidar2img.size(1)
    print(f'num_cam: {num_cam}')

    # Create a new dimension in dimension 2
    # 4 x 1 x 1 x 10 x 4
    reference_points = reference_points.view(D, B, 1, num_query, 4)
    print(reference_points.shape)
    # Repeat num_cam: 8 times in the new dimension 2
    # 4 x 1 x 8 x 10 x 4
    reference_points = reference_points.repeat(1, 1, num_cam, 1, 1)
    print(reference_points.shape)
    # Add a dimension after the last dimension
    # D x B x num_cam(1 -> num_cam) x num_query x (3 + 1) x 1
    # 4 x 1 x 8 x 10 x 4 x 1
    reference_points = reference_points.unsqueeze(-1)
    print(reference_points.shape)

    # 1 x 1 x 8 x 1 x 4 x 4
    lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4)
    print(lidar2img.shape)
    # D(1 -> D) x B x num_cam x num_query(1 -> num_query) x 4 x 4
    # 4 x 1 x 8 x 10 x 4 x 4
    lidar2img = lidar2img.repeat(D, 1, 1, num_query, 1, 1)
    print(lidar2img.shape)

    # Transformation for mapping fake-lidar to real-3D by lidar2img transformation matrix in every last-2 dimension
    # (4 x 4) x (4 x 1) => (4 x 1)
    # 4 x 1 x 8 x 10 x 4 x 1
    reference_points_cam = torch.matmul(lidar2img, reference_points)
    print(
        f'lidar2img_shape:\t\t\t{lidar2img.shape}\nreference_points_shape:\t\t{reference_points.shape}\nreference_points_cam_shape:\t{reference_points_cam.shape}')
    # Remove the last unnecessary dimension
    reference_points_cam = reference_points_cam.squeeze(-1)
    print(reference_points_cam.shape)

    # A threshold for preventing divided by 0
    eps = 1e-5

    # Filter mask in Z axis, True if not super tiny value which considered as valid
    bev_mask = (reference_points_cam[..., 2:3] > eps)
    print(bev_mask.shape)

    # Set minimum value for Z axis to eps
    # 4 x 1 x 8 x 10 x 1
    reference_points_cam_with_eps_floor_z = torch.maximum(reference_points_cam[..., 2:3],
                                                          torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    print(reference_points_cam_with_eps_floor_z.shape)
    # Transform 3D to camera coordinate
    # 4 x 1 x 8 x 10 x 2
    reference_points_cam = reference_points_cam[..., 0:2] / reference_points_cam_with_eps_floor_z
    print(reference_points_cam.shape)

    # Transform from camera to image coordinate
    # reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    # reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    # Update mask, add condition that the point should be located inside the image
    bev_mask = (bev_mask
                & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))
    print(bev_mask.shape)

    # NaN to number
    # D x B x num_cam x num_query x 1
    # 4 x 1 x 8 x 10 x 1
    bev_mask = torch.nan_to_num(bev_mask)
    print(bev_mask.shape)

    # D x B x num_cam x num_query x 2(x, y) -> num_cam x B x num_query x D x 2(x, y)
    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    print(reference_points_cam.shape)
    # 8 x 1 x 10 x 4
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
    print(bev_mask.shape)
