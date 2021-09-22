#pragma once

#include "taichi/program/kernel_profiler.h"

TLANG_NAMESPACE_BEGIN

// make sure these metrics can be captured in one pass (no kernal replay)
enum CuptiMetricsDefault {
  CUPTI_METRIC_KERNEL_ELAPSED_CLK_NUMS = 0,
  CUPTI_METRIC_CORE_FREQUENCY_HZS = 1,
  CUPTI_METRIC_GLOBAL_LOAD_BYTES = 2,
  CUPTI_METRIC_GLOBAL_STORE_BYTES = 3,
  CUPTI_METRIC_DEFAULT_TOTAL = 4
};

const std::vector<std::string> MetricListDeafult = {
    "smsp__cycles_elapsed.avg",  // CUPTI_METRIC_KERNEL_ELAPSED_CLK_NUMS
    "smsp__cycles_elapsed.avg.per_second",  // CUPTI_METRIC_CORE_FREQUENCY_HZS
    "dram__bytes_read.sum",                 // CUPTI_METRIC_GLOBAL_LOAD_BYTES
    "dram__bytes_write.sum"                 // CUPTI_METRIC_GLOBAL_STORE_BYTES
};

struct CuptiConfig {
#if defined(TI_WITH_CUDA_TOOLKIT)
  uint32_t num_ranges = 16384;  // max number of kernels traced by CUPTI
  std::vector<std::string> metric_list;
#endif
};

struct CuptiImage {
  std::string chip_name;
  std::vector<uint8_t> config_image;
  std::vector<uint8_t> counter_availability_image;
  std::vector<uint8_t> counter_data_scratch_buffer;
  std::vector<uint8_t> counter_data_image_prefix;
  std::vector<uint8_t> counter_data_image;
};

bool check_cupti_availability();
bool check_cupti_privileges();

class CuptiToolkit {
 public:
  CuptiToolkit();
  ~CuptiToolkit();

  bool init_cupti();
  bool deinit_cupti();
  bool begin_profiling();
  bool end_profiling();
  bool update_record(std::vector<KernelProfileTracedRecord> &traced_records);

 private:
  CuptiConfig cupti_config_;
  CuptiImage cupti_image_;
};

TLANG_NAMESPACE_END
