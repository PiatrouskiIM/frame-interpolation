import math
import itertools
import numpy as np


def _combination_to_indicator(n, combination):
    indicator = np.zeros(n, dtype=bool)
    indicator[np.array(combination)] = True
    return indicator


def build_holes_pattern_for_upsampling(src_fps, tgt_fps, num_of_sec):
    if tgt_fps % src_fps == 0:
        return np.ones(tgt_fps * num_of_sec, dtype=bool)

    MAX_POWER_OF_2_IN_RATE = int(math.log2(tgt_fps / src_fps))

    num_points_covered_by_doubling = 2 ** MAX_POWER_OF_2_IN_RATE * src_fps
    num_of_unhandled_points = tgt_fps - num_points_covered_by_doubling

    pattern = np.ones(num_of_sec * src_fps * 2 ** (MAX_POWER_OF_2_IN_RATE+1), dtype=bool)

    combinations_iterator = itertools.cycle(itertools.combinations(range(num_points_covered_by_doubling),
                                                                   num_of_unhandled_points))
    pattern[np.arange(1, num_of_sec * src_fps * 2 ** (MAX_POWER_OF_2_IN_RATE+1), step=2)] =\
        np.concatenate([_combination_to_indicator(num_points_covered_by_doubling, next(combinations_iterator))
                        for i in range(num_of_sec) ])
    return pattern
