import numpy as np

from .patch_based_transform import PatchBasedTransform, rectangles2slices


class AssembleFromPatches(PatchBasedTransform):
    def __init__(self, size, patch_size=256, margin=10):
        super().__init__(size, patch_size, margin)

    def __call__(self, x, out=None):
        batch_size = x.shape[1]
        if out is None:
            out = np.zeros((batch_size, 3) + tuple(self.size[::-1]))

        for i, (src_slices, tgt_slices) in enumerate(zip(rectangles2slices(self.in_src_rectangles),
                                                         rectangles2slices(self.in_tgt_rectangles))):
            src_x_slice, src_y_slice = src_slices
            tgt_x_slice, tgt_y_slice = tgt_slices

            out[..., tgt_y_slice, tgt_x_slice] = x[i, ..., src_y_slice, src_x_slice]
        return out
