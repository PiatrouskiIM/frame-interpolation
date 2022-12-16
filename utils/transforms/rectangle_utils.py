import cv2
import numpy as np
from typing import List


def arrange_centers_for_cover_by_size(rectangle_size, size: tuple) -> np.ndarray:
    assert rectangle_size[0] > 0 and rectangle_size[1] > 0 and size[0] > 0 and size[1] > 0, \
        "Argument error. Invalid argument, size must be positive."
    size = np.array(size)
    steps_w, steps_h = np.ceil(rectangle_size / size)
    left_top_corners = np.mgrid[:steps_w, :steps_h].transpose(1, 2, 0).reshape(-1, 2) * size
    return left_top_corners + size / 2


def arrange_centers_for_cover(rectangle, size: tuple) -> np.ndarray:
    """
    For the specified rectangular area and size, determines the positions so that rectangles of this size centered at
    these positions do not intersect and cover this specified area.

    Args:
        rectangle: `tuple` of four `int`s in format `(left, top, right, bottom)`.
            The rectangular region that suppose to be covered.
        size: `tuple` of two `int`s in format `(width, height)`. Sizes of covering rectangles.

    Returns:
        np.ndarray of shape (-1, 2). Centers of the rectangles that form disjoint cover of the provided rectangle.
    """
    l, t, r, b = rectangle
    return arrange_centers_for_cover_by_size(rectangle_size=(r - l, b - t), size=size) + np.array([[l, t]])


def get_shifts_to_be_inside(points: np.ndarray, rectangle: tuple) -> np.ndarray:
    """
    Specifies tha array of shift vectors for the input array to the nearest in `l1`-norm points inside the
    specified rectangular area.

    Args:
        points: array of shape `(N, 2)`. Array of points.
        rectangle: `tuple` of four `int`s in format `(left, top, right, bottom)`.
            The rectangular region.

    Returns:
        Array of shift vectors for the input array to the nearest in `l1`-norm points inside the specified rectangular
        area.
    """
    return np.clip(points, a_min=rectangle[:2], a_max=rectangle[2:]) - points


def get_shifts_to_be_inside_by_size(centers, sizes, in_size) -> np.ndarray:
    sizes = np.array(sizes)
    shifted_centers = np.clip(centers, a_min=sizes / 2, a_max=in_size - sizes / 2)
    return shifted_centers - centers


def xy_wh2lt_rb(boxes: np.ndarray) -> np.ndarray:
    xy_, half_wh = boxes[..., [0], :], boxes[..., 1, :] / 2
    return xy_ + np.stack((-half_wh, half_wh), axis=-2)


def xywh2ltrb(boxes: np.ndarray) -> np.ndarray:
    return xy_wh2lt_rb(boxes.reshape(-1, 2, 2)).reshape(-1, 4)


def lt_rb2xy_wh(boxes: np.ndarray) -> np.ndarray:
    lt_, rb_ = boxes[..., [0], :], boxes[..., [1], :]
    return lt_ + (rb_ - lt_) / 2


def ltrb2xywh(boxes: np.ndarray) -> np.ndarray:
    return lt_rb2xy_wh(boxes.reshape(-1, 2, 2)).reshape(-1, 4)


def get_sizes(lt_rb_boxes: np.ndarray) -> np.ndarray:
    return lt_rb_boxes[..., 1, :] - lt_rb_boxes[..., 0, :]


def rectangles2slices(rectangles: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda x: slice(*x), axis=-1, arr=rectangles.transpose(0, 2, 1))


def take_rectangles_from_image(src: np.ndarray, rectangles: np.ndarray) -> List[np.ndarray]:
    return [src[tuple(slices[::-1])] for slices in rectangles2slices(rectangles)]


def assemble_image_from_overlapping_patches(sources, in_src_rects, in_tgt_rects, target=None):
    assert len(in_src_rects) == len(in_tgt_rects), \
        "The number of rectangles in `in_src_rects` and `in_tgt_rects` must match."
    assert (get_sizes(in_src_rects) == get_sizes(in_tgt_rects)).all(), \
        "The size of the rectangles in `in_src_rects` must much the corresponding size of the rectangles in " \
        "`in_tgt_rects`."

    if target is None:
        n_dims = sources[0].shape[2]
        target_size = in_tgt_rects.reshape(-1, 2).max(axis=0)
        target = np.zeros(tuple(target_size[::-1]) + (n_dims,), dtype=np.uint8)

    for src, src_slices, tgt_slices in zip(sources, rectangles2slices(in_src_rects), rectangles2slices(in_tgt_rects)):
        target[tuple(tgt_slices[::-1])] = src[tuple(src_slices[::-1])]
    return target


def build_processing_functions(input_size, patch_size=256, margin=10):
    if isinstance(margin, int):
        margin = (margin, margin)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    input_size, patch_size, margin = np.array(input_size), np.array(patch_size), np.array(margin)

    assert np.all(margin >= 0), "Argument error. Margin cannot be negative."
    assert np.all(2 * margin < patch_size), \
        f"Patches of size {patch_size} and the margin {margin} will have empty interior."

    inner_size = patch_size - 2 * margin
    centers = arrange_centers_for_cover_by_size(rectangle_size=input_size, size=inner_size)
    shifts = get_shifts_to_be_inside_by_size(centers, sizes=patch_size, in_size=input_size)
    is_shifted = (shifts != 0).any(axis=-1).reshape(-1, 1, 1)
    shifted_centers = centers + shifts

    boxes_xy_wh = np.stack((shifted_centers, np.ones_like(shifted_centers) * np.array(patch_size)), axis=1)
    spliter_functor = lambda src: np.stack(take_rectangles_from_image(src, xy_wh2lt_rb(boxes_xy_wh).astype(int)))

    border_rectangle, inner_rectangle = np.array([[[0, 0], patch_size]]), np.array([[margin, margin + inner_size]])
    in_src_rectangles = (is_shifted == False) * inner_rectangle + is_shifted * border_rectangle

    is_shifted = is_shifted.reshape(-1, 1)
    in_tgt_rectangles = np.stack((shifted_centers,
                                  is_shifted * patch_size + (is_shifted == False) * inner_size), axis=1)
    in_tgt_rectangles = xy_wh2lt_rb(in_tgt_rectangles)
    assembler_functor = lambda patches: assemble_image_from_overlapping_patches(patches,
                                                                                in_src_rectangles.astype(int),
                                                                                in_tgt_rectangles.astype(int))
    return spliter_functor, assembler_functor


if __name__ == "__main__":
    im = cv2.imread("/home/ivan/Experiments/FILM/SKINDRED2/a.png")

    patch_size = (256, 256)
    image_size = im.shape[:2][::-1]
    patch_size = np.array(patch_size)

    spliter, assembler = build_processing_functions(image_size, patch_size, (10, 64))

    # spliter = build_processing_functions(image_size, patch_size, (10, 64))
    patches = spliter(im)
    print("")
    image = assembler(patches)

    cv2.imshow("A", image - im)
    cv2.imshow("B", im)
    cv2.waitKey()
