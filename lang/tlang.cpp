#include "tlang.h"
#include <taichi/system/timer.h>
#include <taichi/testing.h>

TC_NAMESPACE_BEGIN

namespace Tlang {

real get_cpu_frequency() {
  static real cpu_frequency = 0;
  if (cpu_frequency == 0) {
    uint64 cycles = Time::get_cycles();
    Time::sleep(1);
    uint64 elapsed_cycles = Time::get_cycles() - cycles;
    auto frequency = real(std::round(elapsed_cycles / 1e8_f64) / 10.0_f64);
    TC_INFO("CPU frequency = {:.1f} GHz ({} cycles per second)", frequency,
            elapsed_cycles);
    cpu_frequency = frequency;
  }
  return cpu_frequency;
}

real default_measurement_time = 1;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second) {
  if (time_second == 0) {
    target();
    return std::numeric_limits<real>::quiet_NaN();
  }
  // first make rough estimate of run time.
  int64 batch_size = 1;
  while (true) {
    float64 t = Time::get_time();
    for (int64 i = 0; i < batch_size; i++) {
      target();
    }
    t = Time::get_time() - t;
    if (t < 0.05 * time_second) {
      batch_size *= 2;
    } else {
      break;
    }
  }

  int64 total_batches = 0;
  float64 start_t = Time::get_time();
  while (Time::get_time() - start_t < time_second) {
    for (int i = 0; i < batch_size; i++) {
      target();
    }
    total_batches += batch_size;
  }
  auto elasped_cycles =
      (Time::get_time() - start_t) * 1e9_f64 * get_cpu_frequency();
  return elasped_cycles / float64(total_batches * elements_per_call);
}

TC_TEST("Adapter") {
  {
    // num_groups, num_inputs, input_group_size, output_group_size
    VV<32, int> input0;
    for (int i = 0; i < 32; i++) {
      input0[i] = i;
    }
    // num_groups, num_inputs, input_group_size, output_group_size
    Adapter<int, 8, 1, 4, 1> adapter;
    adapter.set<0>(input0);
    adapter.shuffle();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 8; j++) {
        TC_CHECK(adapter.get(i)[j] == i + j * 4);
      }
    }
  }
  {
    // num_groups, num_inputs, input_group_size, output_group_size
    Adapter<int, 8, 4, 1, 4> adapter;
    VV<8, int> inputs[4];
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 8; j++) {
        inputs[i][j] = i + j * 4;
      }
      adapter.set(i, inputs[i]);
    }
    adapter.shuffle();
    for (int i = 0; i < 32; i++) {
      TC_CHECK(adapter.get(0)[i] == i);
    }
  }
}
}

TC_NAMESPACE_END
