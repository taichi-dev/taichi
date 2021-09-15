#pragma once

#if defined(TI_WITH_CUDA_TOOLKIT)

#include <cupti_target.h>
#include <cupti_result.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

// Some of the codes are copied from CUPTI/samples/extensions/
// and modified to match Taichi's naming convensions

template <typename T>
class ScopeExit {
 public:
  ScopeExit(T t) : t_(t) {
  }
  ~ScopeExit() {
    t_();
  }
  T t_;
};

template <typename T>
ScopeExit<T> MoveScopeExit(T t) {
  return ScopeExit<T>(t);
};

  // The fellowing macros will be #undef in ./cupti_toolkit.cpp

#define CUPTI_API_CALL(api_func_call)                                      \
  do {                                                                     \
    CUptiResult status = api_func_call;                                    \
    if (status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                  \
      cuptiGetResultString(status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #api_func_call, errstr);                 \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define NVPW_API_CALL(api_func_call)                                       \
  do {                                                                     \
    NVPA_Status status = api_func_call;                                    \
    if (status != NVPA_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
              __FILE__, __LINE__, #api_func_call, status);                 \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define NV_ANONYMOUS_VARIABLE_DIRECT(name, line) name##line

#define NV_ANONYMOUS_VARIABLE_INDIRECT(name, line) \
  NV_ANONYMOUS_VARIABLE_DIRECT(name, line)

#define SCOPE_EXIT(func)                                      \
  const auto NV_ANONYMOUS_VARIABLE_INDIRECT(EXIT, __LINE__) = \
      MoveScopeExit([=]() { func; })

#define RETURN_IF_NVPW_ERROR(retval, actual)                 \
  do {                                                       \
    NVPA_Status status = actual;                             \
    if (NVPA_STATUS_SUCCESS != status) {                     \
      fprintf(stderr, "FAILED: %s with error %s\n", #actual, \
              get_nvpw_result_string(status));               \
      return retval;                                         \
    }                                                        \
  } while (0)

#define NVPW_STATUS_RESULT(status) \
  case status:                     \
    error_msg = #status;           \
    break;

static const char *get_nvpw_result_string(NVPA_Status status) {
  const char *error_msg = NULL;
  switch (status) {
    NVPW_STATUS_RESULT(NVPA_STATUS_ERROR)
    NVPW_STATUS_RESULT(NVPA_STATUS_INTERNAL_ERROR)
    NVPW_STATUS_RESULT(NVPA_STATUS_NOT_INITIALIZED)
    NVPW_STATUS_RESULT(NVPA_STATUS_NOT_LOADED)
    NVPW_STATUS_RESULT(NVPA_STATUS_FUNCTION_NOT_FOUND)
    NVPW_STATUS_RESULT(NVPA_STATUS_NOT_SUPPORTED)
    NVPW_STATUS_RESULT(NVPA_STATUS_NOT_IMPLEMENTED)
    NVPW_STATUS_RESULT(NVPA_STATUS_INVALID_ARGUMENT)
    NVPW_STATUS_RESULT(NVPA_STATUS_INVALID_METRIC_ID)
    NVPW_STATUS_RESULT(NVPA_STATUS_DRIVER_NOT_LOADED)
    NVPW_STATUS_RESULT(NVPA_STATUS_OUT_OF_MEMORY)
    NVPW_STATUS_RESULT(NVPA_STATUS_INVALID_THREAD_STATE)
    NVPW_STATUS_RESULT(NVPA_STATUS_FAILED_CONTEXT_ALLOC)
    NVPW_STATUS_RESULT(NVPA_STATUS_UNSUPPORTED_GPU)
    NVPW_STATUS_RESULT(NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION)
    NVPW_STATUS_RESULT(NVPA_STATUS_OBJECT_NOT_REGISTERED)
    NVPW_STATUS_RESULT(NVPA_STATUS_INSUFFICIENT_PRIVILEGE)
    NVPW_STATUS_RESULT(NVPA_STATUS_INVALID_CONTEXT_STATE)
    NVPW_STATUS_RESULT(NVPA_STATUS_INVALID_OBJECT_STATE)
    NVPW_STATUS_RESULT(NVPA_STATUS_RESOURCE_UNAVAILABLE)
    NVPW_STATUS_RESULT(NVPA_STATUS_DRIVER_LOADED_TOO_LATE)
    NVPW_STATUS_RESULT(NVPA_STATUS_INSUFFICIENT_SPACE)
    NVPW_STATUS_RESULT(NVPA_STATUS_OBJECT_MISMATCH)
    NVPW_STATUS_RESULT(NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED)
    default:
      break;
  }
  return error_msg;
}

// copy from CUPTI/samples/extensions/include/profilerhost_util/Parser.h
inline bool parse_metric_name_string(const std::string &metric_name,
                                     std::string *req_name,
                                     bool *isolated,
                                     bool *keep_instances) {
  std::string &name = *req_name;
  name = metric_name;
  if (name.empty()) {
    return false;
  }

  // boost program_options sometimes inserts a \n between the metric name and a
  // '&' at the end
  size_t pos = name.find('\n');
  if (pos != std::string::npos) {
    name.erase(pos, 1);
  }

  // trim whitespace
  while (name.back() == ' ') {
    name.pop_back();
    if (name.empty()) {
      return false;
    }
  }

  *keep_instances = false;
  if (name.back() == '+') {
    *keep_instances = true;
    name.pop_back();
    if (name.empty()) {
      return false;
    }
  }

  *isolated = true;
  if (name.back() == '$') {
    name.pop_back();
    if (name.empty()) {
      return false;
    }
  } else if (name.back() == '&') {
    *isolated = false;
    name.pop_back();
    if (name.empty()) {
      return false;
    }
  }

  return true;
}

#endif
