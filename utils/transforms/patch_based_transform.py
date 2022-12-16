from utils.transforms.rectangle_utils import *


class PatchBasedTransform:
    def __init__(self, size, patch_size=256, margin=10):
        if isinstance(margin, int):
            margin = (margin, margin)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(size, int):
            size = (size, size)
        input_size, patch_size, margin = np.array(size), np.array(patch_size), np.array(margin)

        assert np.all(margin >= 0), "Argument error. Margin cannot be negative."
        assert np.all(2 * margin < patch_size), \
            f"Patches of size {patch_size} and the margin {margin} will have empty interior."

        self.size = input_size
        inner_size = patch_size - 2 * margin
        centers = arrange_centers_for_cover_by_size(rectangle_size=input_size, size=inner_size)
        shifts = get_shifts_to_be_inside_by_size(centers, sizes=patch_size, in_size=input_size)
        is_shifted = (shifts != 0).any(axis=-1).reshape(-1, 1, 1)
        shifted_centers = centers + shifts

        boxes_xy_wh = np.stack((shifted_centers, np.ones_like(shifted_centers) * np.array(patch_size)), axis=1)

        self.rectangles = xy_wh2lt_rb(boxes_xy_wh).astype(int)

        border_rectangle, inner_rectangle = np.array([[[0, 0], patch_size]]), np.array([[margin, margin + inner_size]])
        in_src_rectangles = (is_shifted == False) * inner_rectangle + is_shifted * border_rectangle

        is_shifted = is_shifted.reshape(-1, 1)
        in_tgt_rectangles = np.stack((shifted_centers,
                                      is_shifted * patch_size + (is_shifted == False) * inner_size), axis=1)
        in_tgt_rectangles = xy_wh2lt_rb(in_tgt_rectangles)

        self.in_src_rectangles = in_src_rectangles.astype(int)
        self.in_tgt_rectangles = in_tgt_rectangles.astype(int)

    # @abs
    def __call__(self, x):
        return x
