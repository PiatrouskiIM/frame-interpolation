import math
import numpy as np
import itertools


def fill_in(film_net_model, number_of_fills, full_pyramid_a, full_pyramid_b):
    if number_of_fills <= 0:
        return [(full_pyramid_a, full_pyramid_b)]

    c = film_net_model.run_on_features(full_pyramid_a, full_pyramid_b)
    full_pyramid_c = film_net_model.extract_features_pyramid(c)

    return fill_in(film_net_model, number_of_fills - 1, full_pyramid_a, full_pyramid_c) + \
           fill_in(film_net_model, number_of_fills - 1, full_pyramid_c, full_pyramid_b)


def _combination_to_indicator(n, combination):
    indicator = np.zeros(n, dtype=bool)
    indicator[np.array(combination)] = True
    return indicator


# TODO: replace with delayed list like object
class HolePatterns:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.combinations_iter = itertools.cycle(itertools.combinations(range(n), k))
        self.cache = _combination_to_indicator(self.n, next(self.combinations_iter))

    def get_n_items(self, n):
        if n <= len(self.cache):
            out = self.cache[:n]
            self.cache = self.cache[n:]
            return out
        else:
            while n > len(self.cache):
                self.cache = np.concatenate((self.cache, _combination_to_indicator(self.n,
                                                                                   next(self.combinations_iter))))
            return self.get_n_items(n)


class HolesPattern2:
    def __int__(self, n, k, n_combinations_in_cache=4):
        self.n = n
        self.k = k
        self.n_combinations_in_cache = n_combinations_in_cache
        self.combinations_iterator, self.cached_positions, self.cache = self.reset_pointer()

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start // self.n, key.stop // self.n
            if start < self.cached_positions[0]:
                self.combinations_iterator, self.cached_positions, self.cache = self.reset_pointer()
            if stop > self.cached_positions[1]:
                pass


            # _start, _stop = start % len(self.combinations_iterator), stop % len(self.combinations_iterator)
            pass

    def reset_pointer(self):
        return itertools.cycle(itertools.combinations(range(self.n), self.k)), \
               (0, self.n_combinations_in_cache), \
               np.concatenate([_combination_to_indicator(self.n, next(self.combinations_iterator))
                               for _ in range(self.n_combinations_in_cache)], axis=0)


class Interpolator:
    def __init__(self, film_net_model, src_fps, tgt_fps):
        self.film_net_model = film_net_model
        self.src_fps, self.tgt_fps = src_fps, tgt_fps
        self.rate = tgt_fps / src_fps
        self.number_of_pure_upsamplings = int(math.log2(self.rate))
        self.doubled_fps = 2 ** self.number_of_pure_upsamplings * src_fps
        self.number_of_unhandled_frames = tgt_fps - 2 ** self.number_of_pure_upsamplings * self.src_fps
        if self.number_of_unhandled_frames:
            self.hole_patterns = HolePatterns(self.doubled_fps, self.number_of_unhandled_frames)
        else:
            self.number_of_pure_upsamplings -= 1
            self.hole_patterns = HolePatterns(self.doubled_fps, self.doubled_fps)

    def __call__(self, x):
        features = self.film_net_model.extract_features_pyramid(x)
        pyramids = fill_in(self.film_net_model,
                           number_of_fills=self.number_of_pure_upsamplings,
                           full_pyramid_a=list(map(lambda x: x[:-1], features)),
                           full_pyramid_b=list(map(lambda x: x[1:], features)))
        pattern = self.hole_patterns.get_n_items(len(pyramids))
        frames = []
        for status, (pyramid_a, pyramid_b) in zip(pattern, pyramids):
            intermediate_frames = []
            if status:
                intermediate_frames = [self.film_net_model.run_on_features(pyramid_a, pyramid_b)]

            frames.extend([pyramid_a[-1][:, :self.film_net_model.out_channels]] +
                          intermediate_frames \
                          + [pyramid_b[-1][:, :self.film_net_model.out_channels]])
            del pyramid_a
        # TODO: last frame is missing for now
        return frames
