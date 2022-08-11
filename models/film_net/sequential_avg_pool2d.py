from torch.nn import Module, ModuleList, AvgPool2d


class SequentialAvgPool2d(Module):
    """Creates an image pyramid by successively applying a 2D averaging layer to the image."""

    def __init__(self, number_of_applications=8, kernel_size=2, **avg_pool_params):
        super(SequentialAvgPool2d, self).__init__()
        self.series_of_avg_pool2d = \
            ModuleList([AvgPool2d(kernel_size=kernel_size, **avg_pool_params) for _ in range(number_of_applications)])

    def forward(self, x):
        out = [x]
        for pool in self.series_of_avg_pool2d:
            out.append(pool(out[-1]))
        return out
