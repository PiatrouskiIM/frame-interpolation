import numpy as np
import torch
from torch.nn import Module, Sequential, AvgPool2d, Conv2d, LeakyReLU, ModuleList


class FeaturePyramidExtractor(Module):

    def __init__(self, max_number_of_levels, in_channels=3, multiplayer=64):
        super(FeaturePyramidExtractor, self).__init__()
        intro = Sequential(Conv2d(in_channels=in_channels, out_channels=multiplayer, kernel_size=3, padding="same"),
                           LeakyReLU(negative_slope=0.2),
                           Conv2d(in_channels=multiplayer, out_channels=multiplayer, kernel_size=3, padding="same"),
                           LeakyReLU(negative_slope=0.2))

        self.series_of_convnets = ModuleList([intro] +
                                             [Sequential(AvgPool2d(kernel_size=2),
                                                         Conv2d(in_channels=multiplayer << i - 1,
                                                                out_channels=multiplayer << i,
                                                                kernel_size=3,
                                                                padding="same"),
                                                         LeakyReLU(negative_slope=0.2),
                                                         Conv2d(in_channels=multiplayer << i,
                                                                out_channels=multiplayer << i,
                                                                kernel_size=3,
                                                                padding="same"),
                                                         LeakyReLU(negative_slope=0.2))
                                              for i in range(1, max_number_of_levels)])
        self.max_number_of_levels = max_number_of_levels
        self.multiplayer = multiplayer

    def forward(self, x, number_of_levels=None):
        if number_of_levels is None:
            number_of_levels = self.max_number_of_levels
        else:
            assert number_of_levels <= self.max_number_of_levels, \
                f'This instance of HierarchicalFeaturePyramidExtractor capable of extracting only ' \
                f'{self.max_number_of_levels} levels.'

        feature_pyramid = []
        for convnet in self.series_of_convnets[:number_of_levels]:
            x = convnet(x)
            feature_pyramid.append(x)
        return feature_pyramid

    def run_levelwise_and_align(self, pyramid, number_of_levels=None):
        if number_of_levels is None:
            number_of_levels = self.max_number_of_levels

        series_of_pyramids = [self.forward(pyramid_level, min(len(pyramid) - i, number_of_levels))
                              for i, pyramid_level in enumerate(pyramid)]

        aligned_series_of_pyramids = [
            [series_of_pyramids[i - j][j] for j in range(min(number_of_levels, i + 1))]
            for i in range(len(pyramid))]

        cat_in_channel_dim = lambda x: torch.cat(x, dim=1)
        return list(map(cat_in_channel_dim, aligned_series_of_pyramids))

    def calculate_output_number_of_channels(self, number_image_pyramid_levels, number_of_levels):
        assert number_image_pyramid_levels >= number_of_levels, f"Incompatible values of parameters. " \
                                                                f"The number of levels in the image pyramid " \
                                                                f" should not be less " \
                                                                f"than the number of pyramid levels that are " \
                                                                f"planned to be extracted from each " \
                                                                f"level of the input pyramid." \
                                                                f"{number_image_pyramid_levels}" \
                                                                f"(number_image_pyramid_levels)<" \
                                                                f"{number_of_levels}(number_of_levels)."
        longest_subpyramid_channels = self.multiplayer << np.arange(number_image_pyramid_levels)
        aligned_pyramid_channels = np.zeros(number_image_pyramid_levels, dtype=np.int32)
        for i in range(number_image_pyramid_levels):
            subpyramid_len = min(number_image_pyramid_levels - i, number_of_levels)
            aligned_pyramid_channels[i:i + subpyramid_len] += longest_subpyramid_channels[:subpyramid_len]
        return aligned_pyramid_channels
