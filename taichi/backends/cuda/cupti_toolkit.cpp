#include "taichi/backends/cuda/cupti_toolkit.h"

TLANG_NAMESPACE_BEGIN

void CuptiToolkit::set_enable() {
  cupti_config_.enable = true;
}

CuptiToolkit::CuptiToolkit() {
  TI_TRACE("CuptiToolkit::CuptiToolkit() ");
}

CuptiToolkit::~CuptiToolkit() {
  end_profiling();
  deinit_cupti();
}

// TODO Next PR
bool CuptiToolkit::init_cupti() {
  return false;
}
bool CuptiToolkit::begin_profiling() {
  return false;
}
bool CuptiToolkit::end_profiling() {
  return false;
}
bool CuptiToolkit::deinit_cupti() {
  return false;
}
bool CuptiToolkit::calculate_metric_values() {
  return false;
}

TLANG_NAMESPACE_END
