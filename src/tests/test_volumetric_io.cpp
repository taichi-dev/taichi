/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/io/io.h>
#include <taichi/visualization/particle_visualization.h>

TC_NAMESPACE_BEGIN

void test_volumetric_io() {
  const int nx = 128, ny = 128, nz = 64, step = 16;
  std::vector<RenderParticle> particles;
  Vector3 center = Vector3(nx, ny, nz) * 0.5_f;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        int flag = (i / step) + (j / step) + (k / step);
        if (flag % 2 == 0) {
          particles.push_back(
              RenderParticle(Vector3(i, j, k) - center, Vector3(1, 1, 1)));
        }
      }
    }
  }
  write_vector_to_disk(&particles, "particles.bin");
}

TC_NAMESPACE_END
