import numpy as np
from .patch_based_transform import PatchBasedTransform, rectangles2slices


class SplitIntoPatches(PatchBasedTransform):
    def __init__(self, size, patch_size=256, margin=10):
        super().__init__(size, patch_size, margin)

    def __call__(self, x):
        return np.stack([x[..., slice_along_y, slice_along_x]
                         for slice_along_x, slice_along_y in rectangles2slices(self.rectangles)], axis=0)
