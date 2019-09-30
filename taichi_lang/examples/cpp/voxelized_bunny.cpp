#include <taichi/lang.h>
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

auto voxel_bunny = [](std::vector<std::string> cli_param) {
  int n = 128, num_ch1 = 1, num_ch2 = 1;
  CoreState::set_trigger_gdb_when_crash(true);
  auto param = parse_param(cli_param);

  auto tex = create_instance<Texture>(
      "mesh", Dict()
                  .set("resolution", Vector3(n))
                  .set("translate", Vector3(0.55, 0.35, 0.47))
                  .set("scale", Vector3(0.5))
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
        if (inside) {
          for (int c = 0; c < num_ch1; c++) {
            in_data[c * n * n * n + k * n * n + j * n + i] = 1.f;
            count++;
          }
        }
      }
    }
  }
  int total = (num_ch1 * n * n * n);
  std::cout << "non_zero:" << count << ", total:" << total << std::endl;
  printf("Sparsity: %.3f%%\n", 100.0f * count / total);

  auto f = fopen("bunny128.bin", "wb");
  fwrite(in_data, sizeof(float), num_ch1 * n * n * n, f);
  fclose(f);
};
TC_REGISTER_TASK(voxel_bunny);

TC_NAMESPACE_END
