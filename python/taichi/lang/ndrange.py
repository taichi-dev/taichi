from taichi.lang.matrix import _IntermediateMatrix


class ndrange:
    def __init__(self, *args):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, list):
                args[i] = tuple(arg)
            if not isinstance(arg, tuple):
                args[i] = (0, arg)
            assert len(args[i]) == 2
        self.bounds = args

        self.dimensions = [None] * len(args)
        for i, bound in enumerate(self.bounds):
            self.dimensions[i] = bound[1] - bound[0]

        self.acc_dimensions = self.dimensions.copy()
        for i in reversed(range(len(self.bounds) - 1)):
            self.acc_dimensions[
                i] = self.acc_dimensions[i] * self.acc_dimensions[i + 1]
        if len(self.acc_dimensions
               ) == 0:  # for the empty case, e.g. ti.ndrange()
            self.acc_dimensions = [1]

    def __iter__(self):
        def gen(d, prefix):
            if d == len(self.bounds):
                yield prefix
            else:
                for t in range(self.bounds[d][0], self.bounds[d][1]):
                    yield from gen(d + 1, prefix + (t, ))

        yield from gen(0, ())

    def grouped(self):
        return GroupedNDRange(self)


class GroupedNDRange:
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        for ind in self.r:
            yield _IntermediateMatrix(len(ind), 1, list(ind))
