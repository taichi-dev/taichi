// debug
// region header
#include <common.h>

#define TLANG_KERNEL

#include "/home/yuanming/repos/taichi/projects/taichi_lang//_tlang_cache//tmp0000.cpp"
using namespace taichi;
using namespace Tlang;
extern "C" void func000001(Context context) {
  auto SNode0075_cache = (SNode0075 *)context.buffers[0];
  // region exterior_shared_variable_begin
  const auto const0000 = vec<int32, 4>({0, 0, 0, 0});

  const auto const0001 = vec<int32, 1>({0});

  // region exterior_loop_begin
  auto SNode0075_loop = 0;
  auto SNode0002_cache = access_SNode0002(SNode0075_cache, SNode0075_loop);
  int SNode0002_loop;
  auto SNode0002_cache_n = SNode0002_cache->get_n();
  for (SNode0002_loop = 0; SNode0002_loop < SNode0002_cache_n;
       SNode0002_loop += 1) {
    int index_SNode0002_0_local =
        (((SNode0002_loop >> 0) & ((1 << 16) - 1)) << 0);
    int index_SNode0002_0_global = 0 | index_SNode0002_0_local;
    int index_SNode0002_1_local = 0;
    int index_SNode0002_1_global = 0 | index_SNode0002_1_local;
    int index_SNode0002_2_local = 0;
    int index_SNode0002_2_global = 0 | index_SNode0002_2_local;
    int index_SNode0002_3_local = 0;
    int index_SNode0002_3_global = 0 | index_SNode0002_3_local;
    // region interior_shared_variable_begin
    vec<float32, 4> sum2(0);
    // region interior_loop_begin
    auto SNode0001_cache = access_SNode0001(SNode0002_cache, SNode0002_loop);
    int SNode0001_loop;
    auto SNode0001_cache_n = SNode0001_cache->get_n();
    for (SNode0001_loop = 0; SNode0001_loop < SNode0001_cache_n;
         SNode0001_loop += 1) {
      int index_SNode0001_0_local = 0;
      int index_SNode0001_0_global =
          index_SNode0002_0_global | index_SNode0001_0_local;
      int index_SNode0001_1_local =
          (((SNode0001_loop >> 0) & ((1 << 6) - 1)) << 0);
      int index_SNode0001_1_global =
          index_SNode0002_1_global | index_SNode0001_1_local;
      int index_SNode0001_2_local = 0;
      int index_SNode0001_2_global =
          index_SNode0002_2_global | index_SNode0001_2_local;
      int index_SNode0001_3_local = 0;
      int index_SNode0001_3_global =
          index_SNode0002_3_global | index_SNode0001_3_local;
      // region body
      auto var153_index = vec<int32, 4>(index_SNode0001_0_global);
      auto var153 = add(var153_index, const0000);
      float32 *var151[4];
      var151[0] =
          access_SNode0050(context.buffers[0], var153.element(0), 0, 0, 0);
      var151[1] =
          access_SNode0051(context.buffers[0], var153.element(1), 0, 0, 0);
      var151[2] =
          access_SNode0052(context.buffers[0], var153.element(2), 0, 0, 0);
      var151[3] =
          access_SNode0050(context.buffers[0], var153.element(3), 0, 0, 0);
      auto var184_index = vec<int32, 1>(index_SNode0001_0_global);
      auto var184 = add(var184_index, const0001);
      auto var185_index = vec<int32, 1>(index_SNode0001_1_global);
      auto var185 = add(var185_index, const0001);
      auto var183_addr = access_SNode0009(context.buffers[0], var184.element(0),
                                          var185.element(0), 0, 0);
      auto var183 = vec<float32, 4>::load(var183_addr);
      auto var160_index = vec<int32, 4>(index_SNode0001_1_global);
      auto var160 = add(var160_index, const0000);
      int32 *var165[4];
      var165[0] = access_SNode0000(context.buffers[0], var153.element(0),
                                   var160.element(0), 0, 0);
      var165[1] = access_SNode0000(context.buffers[0], var153.element(1),
                                   var160.element(1), 0, 0);
      var165[2] = access_SNode0000(context.buffers[0], var153.element(2),
                                   var160.element(2), 0, 0);
      var165[3] = access_SNode0000(context.buffers[0], var153.element(3),
                                   var160.element(3), 0, 0);
      auto var164 = vec<int32, 4>::load(var165);
      float32 *var162[4];
      var162[0] =
          access_SNode0055(context.buffers[0], var164.element(0), 0, 0, 0);
      var162[1] =
          access_SNode0055(context.buffers[0], var164.element(1), 0, 0, 0);
      var162[2] =
          access_SNode0055(context.buffers[0], var164.element(2), 0, 0, 0);
      var162[3] =
          access_SNode0055(context.buffers[0], var164.element(3), 0, 0, 0);
      auto var161 = vec<float32, 4>::load1(var162[0]);
      auto var156 = mul(var183, var161);
      auto var187_index = vec<int32, 1>(index_SNode0001_0_global);
      auto var187 = add(var187_index, const0001);
      auto var188_index = vec<int32, 1>(index_SNode0001_1_global);
      auto var188 = add(var188_index, const0001);
      auto var186_addr = access_SNode0012(context.buffers[0], var187.element(0),
                                          var188.element(0), 0, 0);
      auto var186 = vec<float32, 4>::load(var186_addr);
      auto var174 = vec<int32, 4>::load(var165);
      float32 *var172[4];
      var172[0] =
          access_SNode0056(context.buffers[0], var174.element(0), 0, 0, 0);
      var172[1] =
          access_SNode0056(context.buffers[0], var174.element(1), 0, 0, 0);
      var172[2] =
          access_SNode0056(context.buffers[0], var174.element(2), 0, 0, 0);
      var172[3] =
          access_SNode0056(context.buffers[0], var174.element(3), 0, 0, 0);
      auto var171 = vec<float32, 4>::load1(var172[0]);
      auto var167 = mul(var186, var171);
      auto var155 = add(var156, var167);
      auto var190_index = vec<int32, 1>(index_SNode0001_0_global);
      auto var190 = add(var190_index, const0001);
      auto var191_index = vec<int32, 1>(index_SNode0001_1_global);
      auto var191 = add(var191_index, const0001);
      auto var189_addr = access_SNode0015(context.buffers[0], var190.element(0),
                                          var191.element(0), 0, 0);
      auto var189 = vec<float32, 4>::load(var189_addr);
      auto var182 = vec<int32, 4>::load(var165);
      float32 *var180[4];
      var180[0] =
          access_SNode0057(context.buffers[0], var182.element(0), 0, 0, 0);
      var180[1] =
          access_SNode0057(context.buffers[0], var182.element(1), 0, 0, 0);
      var180[2] =
          access_SNode0057(context.buffers[0], var182.element(2), 0, 0, 0);
      var180[3] =
          access_SNode0057(context.buffers[0], var182.element(3), 0, 0, 0);
      auto var179 = vec<float32, 4>::load1(var180[0]);
      auto var175 = mul(var189, var179);
      auto var154 = add(var155, var175);
      sum2 = add(sum2, var154);
      // region interior_loop_end
    }
    // region interior_shared_variable_end
    sum2[3] = 0;
    auto *reduce_target2 = access_SNode0050(
        context.buffers[0], index_SNode0002_0_global, index_SNode0002_1_global,
        index_SNode0002_2_global, index_SNode0002_3_global);
    *reduce_target2 = reduce_sum(sum2);
    // region exterior_loop_end
  }
  // region tail
}
