import math

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

tile_size = 8  # Size of a tile
width, height = 500, 500  # Size of framebuffer
num_triangles = 60  # Number of random colored triangles to be generated
num_samples_per_pixel = 4  # Number of samples per pixel

num_spp_sqrt = int(math.sqrt(num_samples_per_pixel))

samples = ti.Vector.field(3, dtype=ti.f32, shape=(width, height, num_spp_sqrt, num_spp_sqrt))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))


@ti.data_oriented
class TriangleRasterizer:
    def __init__(self, n):
        self.n = n
        self.A = ti.Vector.field(2, dtype=ti.f32)
        self.B = ti.Vector.field(2, dtype=ti.f32)
        self.C = ti.Vector.field(2, dtype=ti.f32)
        self.c0 = ti.Vector.field(3, dtype=ti.f32)
        self.c1 = ti.Vector.field(3, dtype=ti.f32)
        self.c2 = ti.Vector.field(3, dtype=ti.f32)

        self.vertices = ti.root.dense(ti.i, n).place(self.A, self.B, self.C)
        self.colors = ti.root.dense(ti.i, n).place(self.c0, self.c1, self.c2)

        # Tile-based culling
        self.block_num_triangles = ti.field(
            dtype=ti.i32,
            shape=((width - 1) // tile_size + 1, (height - 1) // tile_size + 1),
        )
        self.block_indicies = ti.field(
            dtype=ti.i32,
            shape=((width - 1) // tile_size + 1, (height - 1) // tile_size + 1, n),
        )

    def set_triangle(self, i, v0, v1, v2, c0, c1, c2):
        self.A[i] = v0
        self.B[i] = v1
        self.C[i] = v2
        self.c0[i] = c0
        self.c1[i] = c1
        self.c2[i] = c2

    @staticmethod
    @ti.func
    def point_in_triangle(P, A, B, C):
        alpha = -(P.x - B.x) * (C.y - B.y) + (P.y - B.y) * (C.x - B.x)
        alpha /= -(A.x - B.x) * (C.y - B.y) + (A.y - B.y) * (C.x - B.x)
        beta = -(P.x - C.x) * (A.y - C.y) + (P.y - C.y) * (A.x - C.x)
        beta /= -(B.x - C.x) * (A.y - C.y) + (B.y - C.y) * (A.x - C.x)
        gamma = 1.0 - alpha - beta
        result = alpha >= 0.0 and alpha <= 1.0 and beta >= 0.0 and beta <= 1.0 and gamma >= 0.0
        return result, alpha, beta, gamma

    @staticmethod
    @ti.func
    def bbox_intersect(A0, A1, B0, B1):
        return B0.x < A1.x and B0.y < A1.y and B1.x > A0.x and B1.y > A0.y

    @ti.kernel
    def tile_culling(self):
        for i, j in self.block_num_triangles:
            idx = 0
            tile_min = ti.Vector([i * tile_size, j * tile_size])
            tile_max = ti.Vector([(i + 1) * tile_size, (j + 1) * tile_size])
            for t in range(self.n):
                A, B, C = self.A[t], self.B[t], self.C[t]
                tri_min = ti.min(A, ti.min(B, C))
                tri_max = ti.max(A, ti.max(B, C))
                if self.bbox_intersect(tile_min, tile_max, tri_min, tri_max):
                    self.block_indicies[i, j, idx] = t
                    idx = idx + 1
            self.block_num_triangles[i, j] = idx

    @ti.kernel
    def rasterize(self):
        for i, j in pixels:
            for k in range(self.block_num_triangles[i // tile_size, j // tile_size]):
                idx = self.block_indicies[i // tile_size, j // tile_size, k]
                A, B, C = self.A[idx], self.B[idx], self.C[idx]
                c0, c1, c2 = self.c0[idx], self.c1[idx], self.c2[idx]

                for subi, subj in ti.ndrange(num_spp_sqrt, num_spp_sqrt):
                    P = ti.Vector(
                        [
                            i + (subi + 0.5) / num_spp_sqrt,
                            j + (subj + 0.5) / num_spp_sqrt,
                        ]
                    )
                    result, alpha, beta, gamma = self.point_in_triangle(P, A, B, C)

                    if result:
                        interpolated_color = c0 * alpha + c1 * beta + c2 * gamma
                        samples[i, j, subi, subj] = interpolated_color

            samples_sum = ti.Vector([0.0, 0.0, 0.0])
            for subi, subj in ti.ndrange(num_spp_sqrt, num_spp_sqrt):
                samples_sum += samples[i, j, subi, subj]
            pixels[i, j] = samples_sum / num_samples_per_pixel


def main():
    gui = ti.GUI("Rasterizer", res=(width, height))

    triangles = TriangleRasterizer(num_triangles)

    i = 0
    while gui.running:
        # Set a triangle to a new random triangle
        triangles.set_triangle(
            i % num_triangles,
            ti.Vector(np.random.rand(2) * [width, height]),
            ti.Vector(np.random.rand(2) * [width, height]),
            ti.Vector(np.random.rand(2) * [width, height]),
            ti.Vector(np.random.rand(3)),
            ti.Vector(np.random.rand(3)),
            ti.Vector(np.random.rand(3)),
        )
        i = i + 1

        samples.fill(ti.Vector([1.0, 1.0, 1.0]))
        pixels.fill(ti.Vector([1.0, 1.0, 1.0]))
        triangles.tile_culling()
        triangles.rasterize()
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
