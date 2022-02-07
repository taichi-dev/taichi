#pragma once

#include "taichi/program/kernel_profiler.h"

TLANG_NAMESPACE_BEGIN

struct CuptiConfig {
#if defined(TI_WITH_CUDA_TOOLKIT)
  uint32_t num_ranges = 1048576;  // max number of kernels traced by CUPTI
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
  bool update_record(uint32_t records_size_after_sync,
                     std::vector<KernelProfileTracedRecord> &traced_records);
  void reset_metrics(const std::vector<std::string> &metrics);
  void set_status(bool enable);

 private:
  bool enabled_{false};
  CuptiConfig cupti_config_;
  CuptiImage cupti_image_;
};

TLANG_NAMESPACE_END
