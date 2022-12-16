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
"""The final fusion stage for the film_net frame interpolator.

The inputs to this module are the warped input images, image
features and flow fields, all aligned to the target frame (often
midway point between the two original inputs). The output is the
final image. FILM has no explicit occlusion handling -- instead
using the above mentioned information this module automatically
decides how to best blend the inputs together to produce content
in areas where the pixels can only be borrowed from one of the
inputs.

Similarly, this module also decides on how much to blend in each
input in case of fractional timestep that is not at the halfway
point. For example, if the two inputs images are at t=0 and t=1,
and we were to synthesize a frame at t=0.1, it often makes most
sense to favor the first input. However, this is not always the
case -- in particular in occluded pixels.
The architecture of the Fusion module follows U-net [1]
architecture's decoder side, e.g. each pyramid level consists of
concatenation with upsampled coarser level output, and two 3x3
convolutions.

The upsampling is implemented as 'resize convolution', e.g.
nearest neighbor upsampling followed by 2x2 convolution as
explained in [2]. The classic U-net uses max-pooling which has a
tendency to create checkerboard artifacts.

[1] Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015,
    https://arxiv.org/pdf/1505.04597.pdf
[2] https://distill.pub/2016/deconv-checkerboard/
"""

import torch
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, ModuleList, Upsample


class PyramidFusion(Module):

    def __init__(self, in_channels_list, out_channels=3, m=3, multiplier=64):
        super(PyramidFusion, self).__init__()
        self.number_of_levels = len(in_channels_list)
        # filters_numbers = np.clip(multiplier << np.arange(number_of_levels - 1), a_min=0, a_max=multiplier << m)[::-1]
        filters_numbers = [
            (multiplier << i) if i < m else (multiplier << m) for i in reversed(range(self.number_of_levels - 1))
        ]
        intro = Sequential(Upsample(scale_factor=2),
                           Conv2d(in_channels=in_channels_list[0],
                                  out_channels=filters_numbers[0],
                                  kernel_size=2,
                                  padding='same'))

        self.series_of_upscaling_convnets = \
            ModuleList([intro] +
                       [Sequential(
                           Conv2d(in_channels=filters_number + in_channels,
                                  out_channels=filters_number,
                                  kernel_size=3,
                                  padding='same'),
                           LeakyReLU(negative_slope=0.2),
                           Conv2d(in_channels=filters_number,
                                  out_channels=filters_number,
                                  kernel_size=3,
                                  padding='same'),
                           LeakyReLU(negative_slope=0.2),
                           Upsample(scale_factor=2),
                           Conv2d(in_channels=filters_number,
                                  out_channels=out_channels,
                                  kernel_size=2,
                                  padding='same'))
                           for filters_number, out_channels, in_channels in zip(filters_numbers[:-1],
                                                                                filters_numbers[1:],
                                                                                in_channels_list[1:])])

        self.outro = Sequential(
            Conv2d(in_channels=filters_numbers[-1] + in_channels_list[-1],
                   out_channels=multiplier,
                   kernel_size=3,
                   padding='same'),
            LeakyReLU(negative_slope=0.2),
            Conv2d(in_channels=multiplier, out_channels=multiplier, kernel_size=3, padding='same'),
            LeakyReLU(negative_slope=0.2),
            Conv2d(in_channels=multiplier, out_channels=out_channels, kernel_size=1))

    def forward(self, x):
        assert len(x) == self.number_of_levels, \
            f'Fusion called with different number of pyramid levels ' \
            f'{len(x)} than it was configured for, {self.number_of_levels}.'
        y = x[0]
        for pyramid_level, convnet in zip(x[1:], self.series_of_upscaling_convnets):
            y = convnet(y)
            y = torch.cat((pyramid_level, y), dim=1)
        return self.outro(y)
