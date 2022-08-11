import torch
from additional_functional import dense_image_warp


def devide_by_two(pyramid):
    devide_by_two = lambda x: x / 2
    return list(map(devide_by_two, pyramid))


def move_channels_to_last_dim(pyramid):
    move_channels_to_last_dim = lambda x: torch.permute(x, (0, 2, 3, 1))
    return list(map(move_channels_to_last_dim, pyramid))


def warp(feature_pyramid, flow_pyramid):
    warp = lambda image, flow: dense_image_warp(image, -flow[..., [1, 0]])
    return list(map(lambda x: warp(*x), zip(feature_pyramid, flow_pyramid)))


def concatenate(pyramids, axis=-1):
    concatenate = lambda x: torch.cat(x, dim=axis)
    return list(map(concatenate, zip(*pyramids)))


def build(image, number_of_levels=1):
    pyramid = [image]
    for _ in range(number_of_levels - 1):
        pyramid.append(torch.nn.functional.avg_pool2d(pyramid[-1], kernel_size=2))
    return pyramid


transpose_list_of_list = lambda x: list(map(list, zip(*x)))
