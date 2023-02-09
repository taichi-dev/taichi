#pragma once

typedef enum HIPfunction_attribute_enum {
  HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  HIP_FUNC_ATTRIBUTE_NUM_REGS = 4,
  HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  HIP_FUNC_ATTRIBUTE_MAX
} HIPfunction_attribute;