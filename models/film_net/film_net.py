import numpy as np
from torch.nn import Module
from utils import pyramid_utils as pu
from .feature_pyramid_extractor import FeaturePyramidExtractor
from .pyramid_fusion import PyramidFusion
from .sequential_avg_pool2d import SequentialAvgPool2d
from .hierarchical_flow_estimator import HierarchicalResidualFlowEstimator, turn_residual_flow_into_flow_pyramid

from torchvision.models._api import WeightsEnum, Weights


class FILMNet(Module):
    """In-between frame interpolator."""

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 number_image_pyramid_levels=7,
                 number_features_pyramid_levels=4,
                 filters_multiplier=64,
                 flow_convs_list=(3, 3, 3, 3),
                 flow_convs_numbers=(256, 128, 64, 32),
                 number_significant_flow_levels=5):
        super(FILMNet, self).__init__()
        assert number_image_pyramid_levels >= number_significant_flow_levels, \
            "Argument error. The `number_significant_flow_levels` parameter controls how many levels in the feature " \
            "pyramid will be kept and this number cannot exceed the total number of levels specified by the " \
            "`number_image_pyramid_levels` parameter. "
        assert len(flow_convs_list) == len(flow_convs_numbers), \
            f"Parameters `flow_convs_list` and `flow_convs_numbers` should have same length. " \
            f"{len(flow_convs_list)}!={len(flow_convs_numbers)}"
        assert len(flow_convs_list) >= number_features_pyramid_levels, \
            "Argument error. The `number_features_pyramid_levels` parameter affects the number of feature levels " \
            "of different sizes in the intermediate pyramid. By design, at least one flow conv should be provided " \
            "for each size."

        self.in_channels, self.out_channels = in_channels, out_channels
        self.number_image_pyramid_levels = number_image_pyramid_levels
        self.number_features_pyramid_levels = number_features_pyramid_levels
        self.m = number_significant_flow_levels

        self.sequential_avg_pool2d = SequentialAvgPool2d(number_image_pyramid_levels - 1)
        self.feature_extractor = FeaturePyramidExtractor(in_channels=in_channels,
                                                         multiplayer=filters_multiplier,
                                                         max_number_of_levels=number_features_pyramid_levels)
        out_channels_list = \
            self.feature_extractor.calculate_output_number_of_channels(number_image_pyramid_levels,
                                                                       number_features_pyramid_levels)[::-1]

        get_flow_estimator_config = lambda params: dict(zip(["in_channels", "num_convs", "num_filters"], params))
        self.flow_estimator_configs = \
            list(map(get_flow_estimator_config,
                     zip(2 * out_channels_list[-len(flow_convs_list):], flow_convs_list, flow_convs_numbers)))
        self.residual_flow_estimator = HierarchicalResidualFlowEstimator(self.flow_estimator_configs)

        self.pyramid_fusion_module = PyramidFusion(in_channels_list=2 * out_channels_list[-self.m:] + 10,
                                                   out_channels=out_channels,
                                                   multiplier=filters_multiplier)

    def forward(self, x, y):
        pyramid_x, pyramid_y = self.sequential_avg_pool2d(x), self.sequential_avg_pool2d(y)  # biggest first
        feature_pyramid_x, feature_pyramid_y = self.feature_extractor.run_levelwise_and_align(pyramid_x), \
            self.feature_extractor.run_levelwise_and_align(pyramid_y)

        pyramid_x, pyramid_y = pyramid_x[::-1], pyramid_y[::-1]
        feature_pyramid_x, feature_pyramid_y = feature_pyramid_x[::-1], feature_pyramid_y[::-1]

        residual_pyramid_xy, residual_pyramid_yx = self.residual_flow_estimator(feature_pyramid_x, feature_pyramid_y), \
            self.residual_flow_estimator(feature_pyramid_y, feature_pyramid_x)
        flow_pyramid_xy, flow_pyramid_yx = turn_residual_flow_into_flow_pyramid(residual_pyramid_xy), \
            turn_residual_flow_into_flow_pyramid(residual_pyramid_yx)

        flow_pyramid_xy, flow_pyramid_yx = pu.devide_by_two(flow_pyramid_xy), pu.devide_by_two(flow_pyramid_yx)

        flow_pyramid_xy, flow_pyramid_yx = flow_pyramid_xy[-self.m:], flow_pyramid_yx[-self.m:]
        pyramid_x, pyramid_y = pu.concatenate((pyramid_x[-self.m:], feature_pyramid_x[-self.m:]), axis=1), \
            pu.concatenate((pyramid_y[-self.m:], feature_pyramid_y[-self.m:]), axis=1)

        warped_pyramid_x, warped_pyramid_y = pu.warp(pyramid_x, pu.move_channels_to_last_dim(flow_pyramid_yx)), \
            pu.warp(pyramid_y, pu.move_channels_to_last_dim(flow_pyramid_xy))
        aligned_pyramid = pu.concatenate((warped_pyramid_x, warped_pyramid_y, flow_pyramid_yx, flow_pyramid_xy), axis=1)
        return self.pyramid_fusion_module(aligned_pyramid)

    def extract_features_pyramid(self, x):
        pyramid = self.sequential_avg_pool2d(x)
        feature_pyramid = self.feature_extractor.run_levelwise_and_align(pyramid)
        return pu.concatenate((pyramid, feature_pyramid), axis=1)[::-1]

    def run_on_features(self, full_pyramid_x, full_pyramid_y):
        feature_pyramid_x, feature_pyramid_y = list(map(lambda x: x[:, self.in_channels:], full_pyramid_x)), \
            list(map(lambda x: x[:, self.in_channels:], full_pyramid_y))
        residual_pyramid_xy, residual_pyramid_yx = self.residual_flow_estimator(feature_pyramid_x, feature_pyramid_y), \
            self.residual_flow_estimator(feature_pyramid_y, feature_pyramid_x)
        flow_pyramid_xy, flow_pyramid_yx = turn_residual_flow_into_flow_pyramid(residual_pyramid_xy), \
            turn_residual_flow_into_flow_pyramid(residual_pyramid_yx)

        flow_pyramid_xy, flow_pyramid_yx = pu.devide_by_two(flow_pyramid_xy), pu.devide_by_two(flow_pyramid_yx)

        flow_pyramid_xy, flow_pyramid_yx = flow_pyramid_xy[-self.m:], flow_pyramid_yx[-self.m:]
        pyramid_x, pyramid_y = full_pyramid_x[-self.m:], full_pyramid_y[-self.m:]

        warped_pyramid_x, warped_pyramid_y = pu.warp(pyramid_x, pu.move_channels_to_last_dim(flow_pyramid_yx)), \
            pu.warp(pyramid_y, pu.move_channels_to_last_dim(flow_pyramid_xy))
        aligned_pyramid = pu.concatenate((warped_pyramid_x, warped_pyramid_y, flow_pyramid_yx, flow_pyramid_xy), axis=1)
        return self.pyramid_fusion_module(aligned_pyramid)


def preprocess(x: np.ndarray):
    x = x[..., ::-1]  # (reorder color channels) BGR -> RGB
    x = x / 255  # (range change) [0, 255] -> [0, 1]
    return x.transpose(0, 3, 1, 2)  # (reshape) B x H x W x C -> B x C x H x W


def postprocessing(x: np.ndarray):
    x = x.transpose(0, 2, 3, 1)  # (reshape) B x C x H x W -> B x H x W x C
    x = np.array(np.clip(x * 255, 0, 255) + 0.5, dtype=np.uint8)  # (range change with rounding) [0, 1] -> [0, 255]
    return x[..., ::-1]  # (change color channels order) RGB -> BGR


# For the Vimeo-90K dataset, each mini-batch contains 256×256 randomly cropped frame
# triplets in the range [0, 1], with the middle being the ground-truth, for a t=0.5
# interpolation. We use a batch size of 8 distributed over 8V100 GPUs. To combat
# over-fitting, we apply data augmentation: Random rotation with [-45◦, 45◦],
# rotation by multiples of 90◦, horizontal flip, and reversing triplets order. We
# optimize our model with Adam [13] using β1=0.9 and β2=0.999, without weight decay.
# We use an initial learning rate of 1e−4 scheduled with exponential decay  rate of
# 0.464, and decay steps of 750K, in a stair-wise manner, for a total of 3M iterations.
class FILMNet_Weights(WeightsEnum):
    Vimeo90K_V1 = Weights(
        url="https://github.com/PiatrouskiIM/film-in-pytorch/releases/download/v1.0.0/film_net.pt",
        transforms=lambda x: x,  # partial(ImageClassification, crop_size=224),
        meta={
            # **_COMMON_META,
            # "num_params": 2542856,
            "recipe": None,
            # "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "_metrics": {
                "Vimeo-90K": {
                    "PSNR": 35.87,
                    "SSIM": 0.968,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using a simple training recipe.
            """,
        },
    )
    DEFAULT = Vimeo90K_V1

    @staticmethod
    def transforms():
        return preprocess

    @staticmethod
    def postprocessing():
        return postprocessing


def film_net(*, weights=None, progress=True, **kwargs):
    """
       Constructs a FILMNet architecture from
       `FILM: Frame Interpolation for Large Motion <https://arxiv.org/abs/2202.04901>`__.
       Args:
           weights (:class:`~torchvision.models.FilmNet`, optional): The
               pretrained weights to use. See
               :class:`~torchvision.models.FilmNet` below for
               more details, and possible values. By default, no pre-trained
               weights are used.
           progress (bool, optional): If True, displays a progress bar of the
               download to stderr. Default is True.
           **kwargs: parameters passed to the ``models.film_net.FeaturePyramidExtractor``,
           ``models.film_net.HierarchicalResidualFlowEstimator``,
           ``models.film_net.PyramidFusion`` inner classes.
       .. autoclass:: torchvision.models.FilmNet
           :members:
       """
    model = FILMNet(in_channels=kwargs.get("in_channels", 3), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model
