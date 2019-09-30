#include <taichi/lang.h>
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

constexpr int n = 128;

constexpr int num_ch1 = 1;

auto bunny_volume_128 = [](std::vector<std::string> cli_param) {
  auto tex = create_instance<Texture>(
      "mesh", Dict()
                  .set("resolution", Vector3(n))
                  .set("translate", Vector3(0.5, 0.5, 0.5))
                  .set("scale", Vector3(1))
                  .set("adaptive", false)
                  .set("filename", "$mpm/bunny_small.obj"));
  float *in_data = new float[num_ch1 * n * n * n];
  memset(in_data, 0, sizeof(float) * num_ch1 * n * n * n);
  int count = 0;
  for (int i = 1; i < n - 2; i++) {
    for (int j = 1; j < n - 2; j++) {
      for (int k = 1; k < n - 2; k++) {
        bool inside = tex->sample((Vector3(0.5f) + Vector3(i, j, k)) *
                                  Vector3(1.0f / (n - 1)))
                          .x > 0.5f;
        // inside = pow<2>(i - n / 2) + pow<2>(k - n / 2) < pow<2>(n / 2) / 2;
        // inside = i < n * 0.8 && j < n * 0.8 && k < n * 0.8;
        if (inside) {
          for (int c = 0; c < num_ch1; c++) {
            in_data[c * n * n * n + k * n * n + j * n + i] = 1.f;
            count++;
          }
        }
      }
    }
  }
  std::cout << "non_zero:" << count << ", total:" << (num_ch1 * n * n * n)
            << std::endl;
  auto f = fopen("bunny_128.bin", "wb");
  fwrite(in_data, sizeof(float), num_ch1 * n * n * n, f);
  fclose(f);
};
TC_REGISTER_TASK(bunny_volume_128);

TC_NAMESPACE_END
