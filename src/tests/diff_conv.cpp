#include <taichi/taichi>
#include <taichi/util.h>

TC_NAMESPACE_BEGIN

auto diff_conv = [](const std::vector<std::string> args) {
  int grid_resolution = 256;
  TC_ASSERT(args.size() == 3);
  float th = std::stof(args[2]);
  TC_P(th);
  auto f = fopen(args[0].c_str(), "rb");

  int n = pow<3>(grid_resolution);
  TC_ASSERT(f);

  std::vector<float32> ret1(n);
  std::fread(ret1.data(), sizeof(float32), ret1.size(), f);
  std::fclose(f);

  f = fopen(args[1].c_str(), "rb");
  TC_ASSERT(f);
  std::vector<float32> ret2(n);
  std::fread(ret2.data(), sizeof(float32), ret2.size(), f);
  std::fclose(f);

  int counter[2] = {0, 0};
  for (int i = 0; i < n; i++) {
    bool same = std::abs(ret1[i] - ret2[i]) < 1e-3f;
    if (same)
      counter[0]++;
    else
      counter[1]++;
  }
  TC_INFO("same {} {}%", counter[0], 100.0f * counter[0] / n);
};

TC_REGISTER_TASK(diff_conv);

TC_NAMESPACE_END
