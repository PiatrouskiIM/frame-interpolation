import torch
from torch import Tensor
import torch.nn.functional


# Pytorch reimplementation of a ``tensorflow_addons.image.dense_image_warp`` function [1].
# The closest alternative from the Pytorch framework is the ``torch.nn.functional.grid_sample`` function.
# (For more detail, see [2]).

# [1]: https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp
# [2]: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
def dense_image_warp(input: Tensor, flow: Tensor) -> Tensor:
    """Image warping using per-pixel flow vectors.

    Apply a non-linear warp to the image, where the warp is specified by a
    dense flow field of offset vectors that define the correspondences of
    pixel values in the output image back to locations in the source image.
    Specifically, the pixel value at `output[b, j, i, c]` is
    `images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c]`.
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    `(b, j - flow[b, j, i, 0], i - flow[b, j, i, 1])`. For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    NOTE: The definition of the flow field above is different from that
    of optical flow. This function expects the negative forward flow from
    output image to source image. Given two images `I_1` and `I_2` and the
    optical flow `F_12` from `I_1` to `I_2`, the image `I_1` can be
    reconstructed by `I_1_rec = dense_image_warp(I_2, -F_12)`.

    Args:
      input: 4-D float `Tensor` with shape :math:`(N, C, H, W)`.
      flow: A 4-D float `Tensor` with shape :math:`(N, H, W, 2)`.

    Returns:
         A 4-D float `Tensor` with shape :math:`(N, C, H, W)`
           and same type as input image.
    """
    b, h, w, _ = flow.size()
    indices = torch.stack(torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w), indexing="ij"), dim=0)
    indices = indices.to(input.device)
    # indices = torch.stack(torch.meshgrid(*map(torch.arange, flow.size()[:-1]), indexing="ij"), dim=0)
    grid, negative_flow_vectors = input.permute(0, 2, 3, 1), -flow.permute(3, 0, 1, 2).float()

    floor = torch.floor(negative_flow_vectors)
    alpha = negative_flow_vectors - floor
    alpha = alpha.unsqueeze(-1)

    indices[1:] = indices[1:] + floor.long()
    indices = torch.concat((indices, indices[1:] + 1), dim=0)
    indices[1:] = torch.clamp(indices[1:],
                              min=torch.Tensor([0]).to(input.device),
                              max=torch.Tensor([h, w, h, w]).reshape(4, 1, 1, 1).to(input.device) - 1)

    top_left, top_right, bottom_left, bottom_right = grid[tuple(indices[[0, 1, 2]])], \
                                                     grid[tuple(indices[[0, 1, 4]])], \
                                                     grid[tuple(indices[[0, 3, 2]])], \
                                                     grid[tuple(indices[[0, 3, 4]])]
    interpolated_top = alpha[1] * (top_right - top_left) + top_left
    interpolated_bottom = alpha[1] * (bottom_right - bottom_left) + bottom_left
    out = alpha[0] * (interpolated_bottom - interpolated_top) + interpolated_top
    return out.permute(0, 3, 1, 2)
