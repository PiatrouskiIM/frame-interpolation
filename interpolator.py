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
    def __init__(self, n, k, n_combinations_to_cache=4):
        self.n, self.k = n, k
        self._combinations_iterator, (self.start, self.stop), self.cache = self._reset_cache(n_combinations_to_cache)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start // self.n, int(math.floor(key.stop / self.n))
            if self.start <= start and stop <= self.stop:
                adopt_range = lambda x: x - self.start * self.n
                start, stop = adopt_range(key.start), adopt_range(key.stop)
                return self.cache[slice(start, stop, key.step)]
            if start < self.start:
                self._combinations_iterator, (self.start, self.stop), self.cache = \
                    self._reset_cache(self.stop - self.start)
                return self[key]
            if stop > self.stop:
                n_combinations_in_cache = self.stop - self.start
                n_query_combinations = stop - start
                n_combinations_to_cache = max(n_combinations_in_cache, n_query_combinations)
                if stop - n_combinations_to_cache <= self.stop:
                    self.start = stop - n_combinations_to_cache
                    self.cache = np.concatenate(
                        [self.cache[-(self.stop - self.start) * self.n:]] +
                        [_combination_to_indicator(self.n, next(self._combinations_iterator))
                         for _ in range(self.stop, stop + 1)], axis=0)
                    self.stop = stop
                else:
                    for _ in range(self.stop, stop - n_combinations_to_cache):
                        next(self._combinations_iter)
                    self.start, self.stop = stop - n_combinations_to_cache, stop
                    self.cache = np.concatenate([_combination_to_indicator(self.n, next(self._combinations_iterator))
                                                 for _ in range(self.start, self.stop + 1)], axis=0)
                return self[key]

    def _reset_cache(self, n_combinations_in_cache=4):
        combinations_iterator = itertools.cycle(itertools.combinations(range(self.n), self.k))
        return combinations_iterator, \
               (0, n_combinations_in_cache), \
               np.concatenate([_combination_to_indicator(self.n, next(combinations_iterator))
                               for _ in range(n_combinations_in_cache)], axis=0)


class Interpolator:
    def __init__(self, film_net_model, src_fps, tgt_fps):
        self.film_net_model = film_net_model
        self.src_fps, self.tgt_fps = src_fps, tgt_fps
        self.rate = tgt_fps / src_fps
        self.number_of_pure_upsamplings = int(math.log2(self.rate))
        self.doubled_fps = 2 ** self.number_of_pure_upsamplings * src_fps
        self.number_of_unhandled_frames = tgt_fps - 2 ** self.number_of_pure_upsamplings * self.src_fps
        if self.number_of_unhandled_frames:
            self.hole_patterns = HolesPattern2(self.doubled_fps, self.number_of_unhandled_frames)
        else:
            self.number_of_pure_upsamplings -= 1
            self.hole_patterns = HolesPattern2(self.doubled_fps, self.doubled_fps)
        self.frame_no = 0
        self.hole_no = 0
        self.diff = 0
    def __call__(self, x, starting_frame=None):
        features = self.film_net_model.extract_features_pyramid(x)
        pyramids = fill_in(self.film_net_model,
                           number_of_fills=self.number_of_pure_upsamplings,
                           full_pyramid_a=list(map(lambda x: x[:-1], features)),
                           full_pyramid_b=list(map(lambda x: x[1:], features)))
        if starting_frame != self.frame_no:
            self.frame_no = starting_frame
            self.hole_no += self.diff
            # print(self.hole_patterns[self.hole_no:self.hole_no+len(pyramids)].astype(int))
        pattern = self.hole_patterns[self.hole_no:self.hole_no+len(pyramids)]
        self.diff = len(pyramids)
        frames = []
        for status, (pyramid_a, pyramid_b) in zip(pattern, pyramids):
            intermediate_frames = []
            if status:
                intermediate_frames = [self.film_net_model.run_on_features(pyramid_a, pyramid_b)]

            frames.extend([pyramid_a[-1][:, :self.film_net_model.out_channels]] +
                          intermediate_frames)
                          # + [pyramid_b[-1][:, :self.film_net_model.out_channels]])
            del pyramid_a
        # TODO: last frame is missing for now
        return frames


if __name__ == "__main__":
    holes = HolesPattern2(n=5, k=3, n_combinations_to_cache=1)
    print(holes[1:23])
    print(holes.start, holes.stop, holes.cache.shape)
    print(holes[1:10])
