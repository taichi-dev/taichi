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
  double sum1 = 0, sum2 = 0;
  float max1 = 0, max2 = 0;
  int total_non_zero = 0;
  for (int i = 0; i < n; i++) {
    sum1 += std::abs(ret1[i]);
    sum2 += std::abs(ret2[i]);
    max1 = std::max(max1, ret1[i]);
    max2 = std::max(max2, ret1[i]);
    bool same = std::abs(ret1[i] - ret2[i]) < th;
    bool non_zero = (ret1[i] != 0) || (ret2[i] != 0);
    total_non_zero += non_zero;
    if (same)
      counter[0]++;

    if (same && total_non_zero)
      counter[1]++;
  }
  TC_INFO("same {} {}%", counter[0], 100.0f * counter[0] / n);
  TC_INFO("non zero same {} {}%", counter[0],
          100.0f * counter[1] / total_non_zero);
  TC_P(sum1 / n);
  TC_P(sum2 / n);
  TC_P(sum1 / total_non_zero);
  TC_P(sum2 / total_non_zero);
  TC_P(max1);
  TC_P(max2);
};

TC_REGISTER_TASK(diff_conv);

TC_NAMESPACE_END
