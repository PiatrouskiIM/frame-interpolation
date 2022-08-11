# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pytorch module for estimating optical flow by a residual flow pyramid.

This approach of estimating optical flow between two images can be traced back
to [1], but is also used by later neural optical flow computation methods such
as SpyNet [2] and PWC-Net [3].

The basic idea is that the optical flow is first estimated in a coarse
resolution, then the flow is upsampled to warp the higher resolution image and
then a residual correction is computed and added to the estimated flow. This
process is repeated in a pyramid on coarse to fine order to successively
increase the resolution of both optical flow and the warped image.

In here, the optical flow predictor is used as an internal component for the
film_net frame interpolator, to warp the two input images into the inbetween,
target frame.

[1] F. Glazer, Hierarchical motion detection. PhD thesis, 1987.
[2] A. Ranjan and M. J. Black, Optical Flow Estimation using a Spatial Pyramid
    Network. 2016
[3] D. Sun X. Yang, M-Y. Liu and J. Kautz, PWC-Net: CNNs for Optical Flow Using
    Pyramid, Warping, and Cost Volume, 2017
"""

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, ModuleList
from additional_functional import dense_image_warp
from typing import List


def get_flow_estimator(in_channels: int, num_filters: int, num_convs: int):
    """Creates a convolutional neural network that will be used to predict the flow.

    Args:
      in_channels: Number of channels in the input signal.
      num_filters: Number of 3x3 convolutions to add.
      num_convs: Number of filters to be set in each added 3x3 convolution.
    Returns:
      A convolutional network consisting of a `num_convs` 3x3 convolutions followed by two 1x1 convolutions.
    """
    intro = [
        Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding='same'),
        LeakyReLU(negative_slope=0.2)
    ]
    activated_conv_list = [
        layer for _ in range(num_convs - 1) for layer in (Conv2d(in_channels=num_filters,
                                                                 out_channels=num_filters,
                                                                 kernel_size=3,
                                                                 padding='same'),
                                                          LeakyReLU(negative_slope=0.2))
    ]
    outro = [
        Conv2d(in_channels=num_filters, out_channels=num_filters // 2, kernel_size=1),
        LeakyReLU(negative_slope=0.2),
        Conv2d(in_channels=num_filters // 2, out_channels=2, kernel_size=1)
    ]
    return Sequential(*(intro + activated_conv_list + outro))


def turn_residual_flow_into_flow_pyramid(pyramid: List[Tensor]) -> List[Tensor]:
    """Cumulative sum of input tensors.

    Args:
      pyramid: List of tensors, each subsequent tensor of which has the same number of channels and twice the
    resolution.
    Returns:
      List of tensors obtained by the cumulative sum of the inputs.
    """
    flow = pyramid[0]
    flow_pyramid = [flow]
    for level in pyramid[1:]:
        flow = torch.nn.functional.interpolate(input=2 * flow, scale_factor=2, mode='bilinear')
        flow = flow + level
        flow_pyramid.append(flow)
    return flow_pyramid


def warp_with_flow(image, flow):
    return dense_image_warp(image, -flow[..., [1, 0]])


class HierarchicalResidualFlowEstimator(Module):
    """Predicts optical flow by coarse-to-fine refinement."""

    def __init__(self, flow_estimators_configs):
        super(HierarchicalResidualFlowEstimator, self).__init__()
        self.series_of_convnets = ModuleList([get_flow_estimator(**config) for config in flow_estimators_configs])

    def forward(self, pyramid_a: List[Tensor], pyramid_b: List[Tensor]) -> List[Tensor]:
        number_shared_levels = len(pyramid_a) - len(self.series_of_convnets)
        flow_estimator_i = 0
        residuals = []
        for level_i, (level_a, level_b) in enumerate(zip(pyramid_a, pyramid_b)):
            if level_i > 0:
                # Upsamples the flow to match the current pyramid level. Also, scales the
                # magnitude by two to reflect the new size.
                v = torch.nn.functional.interpolate(input=2 * v, scale_factor=2, mode='bilinear')
                b_warped_by_v = warp_with_flow(level_b, flow=v.permute(0, 2, 3, 1))
            else:
                v, b_warped_by_v = 0, level_b
            if level_i > number_shared_levels:
                flow_estimator_i = level_i - number_shared_levels
            residuals.append(self.series_of_convnets[flow_estimator_i](torch.cat((level_a, b_warped_by_v), dim=1)))
            v += residuals[-1]
        return residuals
