#pragma once

#if defined(TI_WITH_CUDA_TOOLKIT)

#include <cupti_target.h>
#include <cupti_result.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

template <typename T>

class ScopeExit {
 public:
  ScopeExit(T t) : t(t) {
  }
  ~ScopeExit() {
    t();
  }
  T t;
};

template <typename T>
ScopeExit<T> MoveScopeExit(T t) {
  return ScopeExit<T>(t);
};

#define CUPTI_API_CALL(apiFuncCall)                                        \
  do {                                                                     \
    CUptiResult _status = apiFuncCall;                                     \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #apiFuncCall, errstr);                   \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define NVPW_API_CALL(apiFuncCall)                                         \
  do {                                                                     \
    NVPA_Status _status = apiFuncCall;                                     \
    if (_status != NVPA_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
              __FILE__, __LINE__, #apiFuncCall, _status);                  \
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
              GetNVPWResultString(status));                  \
      return retval;                                         \
    }                                                        \
  } while (0)

static const char *GetNVPWResultString(NVPA_Status status) {
  const char *errorMsg = NULL;
  switch (status) {
    case NVPA_STATUS_ERROR:
      errorMsg = "NVPA_STATUS_ERROR";
      break;
    case NVPA_STATUS_INTERNAL_ERROR:
      errorMsg = "NVPA_STATUS_INTERNAL_ERROR";
      break;
    case NVPA_STATUS_NOT_INITIALIZED:
      errorMsg = "NVPA_STATUS_NOT_INITIALIZED";
      break;
    case NVPA_STATUS_NOT_LOADED:
      errorMsg = "NVPA_STATUS_NOT_LOADED";
      break;
    case NVPA_STATUS_FUNCTION_NOT_FOUND:
      errorMsg = "NVPA_STATUS_FUNCTION_NOT_FOUND";
      break;
    case NVPA_STATUS_NOT_SUPPORTED:
      errorMsg = "NVPA_STATUS_NOT_SUPPORTED";
      break;
    case NVPA_STATUS_NOT_IMPLEMENTED:
      errorMsg = "NVPA_STATUS_NOT_IMPLEMENTED";
      break;
    case NVPA_STATUS_INVALID_ARGUMENT:
      errorMsg = "NVPA_STATUS_INVALID_ARGUMENT";
      break;
    case NVPA_STATUS_INVALID_METRIC_ID:
      errorMsg = "NVPA_STATUS_INVALID_METRIC_ID";
      break;
    case NVPA_STATUS_DRIVER_NOT_LOADED:
      errorMsg = "NVPA_STATUS_DRIVER_NOT_LOADED";
      break;
    case NVPA_STATUS_OUT_OF_MEMORY:
      errorMsg = "NVPA_STATUS_OUT_OF_MEMORY";
      break;
    case NVPA_STATUS_INVALID_THREAD_STATE:
      errorMsg = "NVPA_STATUS_INVALID_THREAD_STATE";
      break;
    case NVPA_STATUS_FAILED_CONTEXT_ALLOC:
      errorMsg = "NVPA_STATUS_FAILED_CONTEXT_ALLOC";
      break;
    case NVPA_STATUS_UNSUPPORTED_GPU:
      errorMsg = "NVPA_STATUS_UNSUPPORTED_GPU";
      break;
    case NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION:
      errorMsg = "NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION";
      break;
    case NVPA_STATUS_OBJECT_NOT_REGISTERED:
      errorMsg = "NVPA_STATUS_OBJECT_NOT_REGISTERED";
      break;
    case NVPA_STATUS_INSUFFICIENT_PRIVILEGE:
      errorMsg = "NVPA_STATUS_INSUFFICIENT_PRIVILEGE";
      break;
    case NVPA_STATUS_INVALID_CONTEXT_STATE:
      errorMsg = "NVPA_STATUS_INVALID_CONTEXT_STATE";
      break;
    case NVPA_STATUS_INVALID_OBJECT_STATE:
      errorMsg = "NVPA_STATUS_INVALID_OBJECT_STATE";
      break;
    case NVPA_STATUS_RESOURCE_UNAVAILABLE:
      errorMsg = "NVPA_STATUS_RESOURCE_UNAVAILABLE";
      break;
    case NVPA_STATUS_DRIVER_LOADED_TOO_LATE:
      errorMsg = "NVPA_STATUS_DRIVER_LOADED_TOO_LATE";
      break;
    case NVPA_STATUS_INSUFFICIENT_SPACE:
      errorMsg = "NVPA_STATUS_INSUFFICIENT_SPACE";
      break;
    case NVPA_STATUS_OBJECT_MISMATCH:
      errorMsg = "NVPA_STATUS_OBJECT_MISMATCH";
      break;
    case NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED:
      errorMsg = "NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED";
      break;
    default:
      break;
  }

  return errorMsg;
}

inline bool ParseMetricNameString(const std::string &metricName,
                                  std::string *reqName,
                                  bool *isolated,
                                  bool *keepInstances) {
  std::string &name = *reqName;
  name = metricName;
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

  *keepInstances = false;
  if (name.back() == '+') {
    *keepInstances = true;
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
