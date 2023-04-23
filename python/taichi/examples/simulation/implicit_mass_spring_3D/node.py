import taichi as ti
import numpy as np


@ti.data_oriented
class NODE:
    def __init__(self, path):
        self.v = []
        self.f = []
        self.s = set()

        with open(path + ".node", "r") as file:
            head = file.readline()
            for line in file:
                data = line.strip().split()
                # 将所有点放入v
                try:
                    self.v.append([float(pos) for pos in data[1:4]])
                except ValueError:
                    pass

        with open(path + ".ele") as file:
            head = file.readline()
            for line in file:
                data = line.strip().split()
                try:
                    self.f += [[int(pos) for pos in data[3:0:-1]]]
                    self.f += [[int(pos) for pos in data[2:5]]]
                    self.f += [[int(pos) for pos in [data[1], data[4], data[3]]]]
                    self.f += [[int(pos) for pos in (data[1:3] + [data[4]])]]
                except ValueError:
                    pass

        self.vn = len(self.v)
        self.fn = len(self.f)

        self.num_particles = ti.field(int, shape=())
        self.num_particles[None] = self.vn
        self.num_surfaces = ti.field(int, shape=())
        self.num_surfaces[None] = self.fn

        self.vertices = ti.Vector.field(3, dtype=float, shape=self.vn)
        self.indices = ti.field(int, shape=self.fn * 3)

        self.vertices.from_numpy(np.array(self.v))
        self.indices.from_numpy(np.array(self.f).flatten())

        self.init_spring()
        self.sn = len(self.s)
        self.num_springs = ti.field(int, shape=())
        self.num_springs[None] = self.sn
        # 弹簧，储存点的索引
        self.springs = ti.Vector.field(2, dtype=int, shape=self.sn)
        self.init_springs_field()

    def init_spring(self):
        for i in range(self.num_surfaces[None]):
            # Edge 1-2
            if self.indices[i * 3] < self.indices[i * 3 + 1]:
                self.s.add((self.indices[i * 3], self.indices[i * 3 + 1]))
            else:
                self.s.add((self.indices[i * 3 + 1], self.indices[i * 3]))
            # Edge 1-3
            if self.indices[i * 3] < self.indices[i * 3 + 2]:
                self.s.add((self.indices[i * 3], self.indices[i * 3 + 2]))
            else:
                self.s.add((self.indices[i * 3 + 2], self.indices[i * 3]))
            # Edge 2-3
            if self.indices[i * 3 + 1] < self.indices[i * 3 + 2]:
                self.s.add((self.indices[i * 3 + 1], self.indices[i * 3 + 2]))
            else:
                self.s.add((self.indices[i * 3 + 2], self.indices[i * 3 + 1]))

    def init_springs_field(self):
        i = 0
        for p_pair in self.s:
            self.springs[i][0] = p_pair[0]
            self.springs[i][1] = p_pair[1]
            i += 1
