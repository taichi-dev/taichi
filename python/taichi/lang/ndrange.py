import taichi as ti


class ndrange:
    def __init__(self, *args):
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], list):
                args[i] = tuple(args[i])
            if not isinstance(args[i], tuple):
                args[i] = (0, args[i])
            assert len(args[i]) == 2
        self.bounds = args

        self.dimensions = [None] * len(args)
        for i in range(len(self.bounds)):
            self.dimensions[i] = self.bounds[i][1] - self.bounds[i][0]

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
            yield ti.Vector(list(ind), keep_raw=True)
