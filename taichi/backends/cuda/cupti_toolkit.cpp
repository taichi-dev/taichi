#include "taichi/backends/cuda/cupti_toolkit.h"
#include "taichi/backends/cuda/cuda_context.h"

// move from cupti_toolkit.h
// avoid exposing these headers
#if defined(TI_WITH_CUDA_TOOLKIT)
#include <cupti_target.h>
#include <cupti_result.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#endif

TLANG_NAMESPACE_BEGIN

// Make sure these metrics can be captured in one pass (no kernal replay)
// Metrics for calculating the kernel elapsed time are collected by default.
enum class CuptiMetricsDefault : uint {
  CUPTI_METRIC_KERNEL_ELAPSED_CLK_NUMS = 0,
  CUPTI_METRIC_CORE_FREQUENCY_HZS = 1,
  CUPTI_METRIC_DEFAULT_TOTAL = 2
};

[[maybe_unused]] constexpr const char *MetricListDefault[] = {
    "smsp__cycles_elapsed.avg",  // CUPTI_METRIC_KERNEL_ELAPSED_CLK_NUMS
    "smsp__cycles_elapsed.avg.per_second",  // CUPTI_METRIC_CORE_FREQUENCY_HZS
};

bool check_cupti_availability() {
  void *device;
  int cc_major;
  CUDADriver::get_instance().device_get(&device, 0);
  CUDADriver::get_instance().device_get_attribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (cc_major < 7) {
    TI_WARN(
        "CUPTI profiler APIs unsupported on Device with compute capability < "
        "7.0 , fallback to default kernel profiler");
    TI_WARN(
        "See also: "
        "https://docs.taichi-lang.org/docs/profiler");
    return false;
  }
  return true;
}

bool check_cupti_privileges() {
#if defined(TI_WITH_CUDA_TOOLKIT)

#define TEMP_CUPTI_API_CALL(api_func_call)                                 \
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

  CUpti_Profiler_Initialize_Params profiler_initialize_params = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  TEMP_CUPTI_API_CALL(cuptiProfilerInitialize(&profiler_initialize_params));
  TI_TRACE("cuptiProfilerInitialized");

  int device_num = 0;  // TODO
  // CUDADriver::get_instance_without_context().init(0);
  CUpti_Device_GetChipName_Params get_chip_name_params = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  get_chip_name_params.deviceIndex = device_num;
  TEMP_CUPTI_API_CALL(cuptiDeviceGetChipName(&get_chip_name_params));
  std::string chip_name = get_chip_name_params.pChipName;

  CUpti_Profiler_GetCounterAvailability_Params get_counter_availability_params =
      {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  get_counter_availability_params.ctx =
      (CUcontext)(CUDAContext::get_instance().get_context());
  TEMP_CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&get_counter_availability_params));

#undef TEMP_CUPTI_API_CALL

  std::vector<uint8_t> counter_availability_image;
  counter_availability_image.clear();
  counter_availability_image.resize(
      get_counter_availability_params.counterAvailabilityImageSize);
  get_counter_availability_params.pCounterAvailabilityImage =
      counter_availability_image.data();
  CUptiResult status =
      cuptiProfilerGetCounterAvailability(&get_counter_availability_params);

  if (status == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) {
    TI_WARN(
        "function cuptiProfilerInitialize failed with error : "
        "CUPTI_ERROR_INSUFFICIENT_PRIVILEGES");
    TI_WARN("fallback to default kernel profiler : cuEvent");
    TI_WARN(
        "=================================================================");
    TI_WARN("Add `option nvidia NVreg_RestrictProfilingToAdminUsers=0`");
    TI_WARN("to /etc/modprobe.d/nvidia-kernel-common.conf");
    TI_WARN("run `update-initramfs -u`");
    TI_WARN("then `reboot` should resolve the permission issue.");
    TI_WARN(
        "=================================================================");
    TI_WARN(
        "See also: "
        "https://docs.taichi-lang.org/docs/profiler");
    return false;
  }
  // For other errors , CuptiToolkit::init_cupti() will send error message.
  return true;
#else
  return false;
#endif
}

#if defined(TI_WITH_CUDA_TOOLKIT)

// Some of the codes are copied from CUPTI/samples/extensions/
// and modified to match Taichi's naming conventions
template <typename T>
class ScopeExit {
 public:
  ScopeExit(T t) : t_(t) {
  }
  ~ScopeExit() {
    t_();
  }

 private:
  T t_;
};

template <typename T>
ScopeExit<T> MoveScopeExit(T t) {
  return ScopeExit<T>(t);
};

#define NV_ANONYMOUS_VARIABLE_DIRECT(name, line) name##line

#define NV_ANONYMOUS_VARIABLE_INDIRECT(name, line) \
  NV_ANONYMOUS_VARIABLE_DIRECT(name, line)

#define SCOPE_EXIT(func)                                      \
  const auto NV_ANONYMOUS_VARIABLE_INDIRECT(EXIT, __LINE__) = \
      MoveScopeExit([=]() { func; })

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

#define RETURN_IF_NVPW_ERROR(retval, actual)                 \
  do {                                                       \
    NVPA_Status status = actual;                             \
    if (NVPA_STATUS_SUCCESS != status) {                     \
      fprintf(stderr, "FAILED: %s with error %s\n", #actual, \
              get_nvpw_result_string(status));               \
      return retval;                                         \
    }                                                        \
  } while (0)

static const char *get_nvpw_result_string(NVPA_Status status) {
#define NVPW_STATUS_RESULT(status) \
  case status:                     \
    error_msg = #status;           \
    break;

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
#undef NVPW_STATUS_RESULT
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

#if CUDA_VERSION >= 11040

// copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool get_raw_metric_requests(
    std::string chip_name,
    const std::vector<std::string> &metric_names,
    std::vector<NVPA_RawMetricRequest> &raw_metric_requests,
    const uint8_t *p_counter_availability_image) {
  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params
      calculate_scratch_buffer_size_param = {
          NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  calculate_scratch_buffer_size_param.pChipName = chip_name.c_str();
  calculate_scratch_buffer_size_param.pCounterAvailabilityImage =
      p_counter_availability_image;
  RETURN_IF_NVPW_ERROR(false,
                       NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(
                           &calculate_scratch_buffer_size_param));

  std::vector<uint8_t> scratch_buffer(
      calculate_scratch_buffer_size_param.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params
      metric_evaluator_initialize_params = {
          NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  metric_evaluator_initialize_params.scratchBufferSize = scratch_buffer.size();
  metric_evaluator_initialize_params.pScratchBuffer = scratch_buffer.data();
  metric_evaluator_initialize_params.pChipName = chip_name.c_str();
  metric_evaluator_initialize_params.pCounterAvailabilityImage =
      p_counter_availability_image;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(
                                  &metric_evaluator_initialize_params));
  NVPW_MetricsEvaluator *metric_evaluator =
      metric_evaluator_initialize_params.pMetricsEvaluator;

  bool isolated = true;
  bool keep_instances = true;
  std::vector<const char *> raw_metric_names;
  for (auto &metric_name : metric_names) {
    std::string req_name;
    parse_metric_name_string(metric_name, &req_name, &isolated,
                             &keep_instances);
    keep_instances = true;
    NVPW_MetricEvalRequest metric_eval_request;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
        convert_metric_to_eval_request = {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convert_metric_to_eval_request.pMetricsEvaluator = metric_evaluator;
    convert_metric_to_eval_request.pMetricName = req_name.c_str();
    convert_metric_to_eval_request.pMetricEvalRequest = &metric_eval_request;
    convert_metric_to_eval_request.metricEvalRequestStructSize =
        NVPW_MetricEvalRequest_STRUCT_SIZE;
    RETURN_IF_NVPW_ERROR(
        false, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
                   &convert_metric_to_eval_request));

    std::vector<const char *> raw_dependencies;
    NVPW_MetricsEvaluator_GetMetricRawDependencies_Params
        get_metric_raw_dependencies_parms = {
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
    get_metric_raw_dependencies_parms.pMetricsEvaluator = metric_evaluator;
    get_metric_raw_dependencies_parms.pMetricEvalRequests =
        &metric_eval_request;
    get_metric_raw_dependencies_parms.numMetricEvalRequests = 1;
    get_metric_raw_dependencies_parms.metricEvalRequestStructSize =
        NVPW_MetricEvalRequest_STRUCT_SIZE;
    get_metric_raw_dependencies_parms.metricEvalRequestStrideSize =
        sizeof(NVPW_MetricEvalRequest);
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricRawDependencies(
                                    &get_metric_raw_dependencies_parms));
    raw_dependencies.resize(
        get_metric_raw_dependencies_parms.numRawDependencies);
    get_metric_raw_dependencies_parms.ppRawDependencies =
        raw_dependencies.data();
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricRawDependencies(
                                    &get_metric_raw_dependencies_parms));

    for (size_t i = 0; i < raw_dependencies.size(); ++i) {
      raw_metric_names.push_back(raw_dependencies[i]);
    }
  }

  for (auto &raw_metric_name : raw_metric_names) {
    NVPA_RawMetricRequest metric_request = {
        NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
    metric_request.pMetricName = raw_metric_name;
    metric_request.isolated = isolated;
    metric_request.keepInstances = keep_instances;
    raw_metric_requests.push_back(metric_request);
  }

  NVPW_MetricsEvaluator_Destroy_Params metric_evaluator_destroy_params = {
      NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
  metric_evaluator_destroy_params.pMetricsEvaluator = metric_evaluator;
  RETURN_IF_NVPW_ERROR(
      false, NVPW_MetricsEvaluator_Destroy(&metric_evaluator_destroy_params));
  return true;
}

// copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool get_config_image(std::string chip_name,
                      const std::vector<std::string> &metric_names,
                      std::vector<uint8_t> &config_image,
                      const uint8_t *p_counter_availability_image) {
  std::vector<NVPA_RawMetricRequest> raw_metric_requests;
  get_raw_metric_requests(chip_name, metric_names, raw_metric_requests,
                          p_counter_availability_image);

  NVPW_CUDA_RawMetricsConfig_Create_V2_Params raw_metrics_config_create_params =
      {NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE};
  raw_metrics_config_create_params.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
  raw_metrics_config_create_params.pChipName = chip_name.c_str();
  raw_metrics_config_create_params.pCounterAvailabilityImage =
      p_counter_availability_image;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_RawMetricsConfig_Create_V2(
                                  &raw_metrics_config_create_params));
  NVPA_RawMetricsConfig *p_raw_metrics_config =
      raw_metrics_config_create_params.pRawMetricsConfig;

  if (p_counter_availability_image) {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params
        set_counter_availability_params = {
            NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
    set_counter_availability_params.pRawMetricsConfig = p_raw_metrics_config;
    set_counter_availability_params.pCounterAvailabilityImage =
        p_counter_availability_image;
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_SetCounterAvailability(
                                    &set_counter_availability_params));
  }

  NVPW_RawMetricsConfig_Destroy_Params raw_metrics_config_destroy_params = {
      NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
  raw_metrics_config_destroy_params.pRawMetricsConfig = p_raw_metrics_config;
  SCOPE_EXIT([&]() {
    NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params
                                       *)&raw_metrics_config_destroy_params);
  });

  NVPW_RawMetricsConfig_BeginPassGroup_Params begin_pass_group_params = {
      NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
  begin_pass_group_params.pRawMetricsConfig = p_raw_metrics_config;
  RETURN_IF_NVPW_ERROR(
      false, NVPW_RawMetricsConfig_BeginPassGroup(&begin_pass_group_params));

  NVPW_RawMetricsConfig_AddMetrics_Params add_metrics_params = {
      NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
  add_metrics_params.pRawMetricsConfig = p_raw_metrics_config;
  add_metrics_params.pRawMetricRequests = raw_metric_requests.data();
  add_metrics_params.numMetricRequests = raw_metric_requests.size();
  RETURN_IF_NVPW_ERROR(false,
                       NVPW_RawMetricsConfig_AddMetrics(&add_metrics_params));

  NVPW_RawMetricsConfig_EndPassGroup_Params end_pass_group_params = {
      NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
  end_pass_group_params.pRawMetricsConfig = p_raw_metrics_config;
  RETURN_IF_NVPW_ERROR(
      false, NVPW_RawMetricsConfig_EndPassGroup(&end_pass_group_params));

  NVPW_RawMetricsConfig_GenerateConfigImage_Params
      generate_config_image_params = {
          NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE};
  generate_config_image_params.pRawMetricsConfig = p_raw_metrics_config;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GenerateConfigImage(
                                  &generate_config_image_params));

  NVPW_RawMetricsConfig_GetConfigImage_Params get_config_image_params = {
      NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE};
  get_config_image_params.pRawMetricsConfig = p_raw_metrics_config;
  get_config_image_params.bytesAllocated = 0;
  get_config_image_params.pBuffer = NULL;
  RETURN_IF_NVPW_ERROR(
      false, NVPW_RawMetricsConfig_GetConfigImage(&get_config_image_params));

  config_image.resize(get_config_image_params.bytesCopied);
  get_config_image_params.bytesAllocated = config_image.size();
  get_config_image_params.pBuffer = config_image.data();
  RETURN_IF_NVPW_ERROR(
      false, NVPW_RawMetricsConfig_GetConfigImage(&get_config_image_params));

  return true;
}

// copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool get_counter_data_prefix_image(
    std::string chip_name,
    const std::vector<std::string> &metric_names,
    std::vector<uint8_t> &counter_data_image_prefix,
    const uint8_t *p_counter_availability_image) {
  std::vector<NVPA_RawMetricRequest> raw_metric_requests;
  get_raw_metric_requests(chip_name, metric_names, raw_metric_requests,
                          p_counter_availability_image);

  NVPW_CUDA_CounterDataBuilder_Create_Params
      counter_data_builder_create_params = {
          NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE};
  counter_data_builder_create_params.pChipName = chip_name.c_str();
  counter_data_builder_create_params.pCounterAvailabilityImage =
      p_counter_availability_image;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_CounterDataBuilder_Create(
                                  &counter_data_builder_create_params));

  NVPW_CounterDataBuilder_Destroy_Params counter_data_builder_destroy_params = {
      NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
  counter_data_builder_destroy_params.pCounterDataBuilder =
      counter_data_builder_create_params.pCounterDataBuilder;
  SCOPE_EXIT([&]() {
    NVPW_CounterDataBuilder_Destroy(
        (NVPW_CounterDataBuilder_Destroy_Params
             *)&counter_data_builder_destroy_params);
  });

  NVPW_CounterDataBuilder_AddMetrics_Params add_metrics_params = {
      NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
  add_metrics_params.pCounterDataBuilder =
      counter_data_builder_create_params.pCounterDataBuilder;
  add_metrics_params.pRawMetricRequests = raw_metric_requests.data();
  add_metrics_params.numMetricRequests = raw_metric_requests.size();
  RETURN_IF_NVPW_ERROR(false,
                       NVPW_CounterDataBuilder_AddMetrics(&add_metrics_params));

  NVPW_CounterDataBuilder_GetCounterDataPrefix_Params
      get_counter_dataPrefix_params = {
          NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
  get_counter_dataPrefix_params.pCounterDataBuilder =
      counter_data_builder_create_params.pCounterDataBuilder;
  get_counter_dataPrefix_params.bytesAllocated =
      0;  // size_t counterDataPrefixSize = 0
  get_counter_dataPrefix_params.pBuffer = NULL;
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(
                                  &get_counter_dataPrefix_params));

  counter_data_image_prefix.resize(get_counter_dataPrefix_params.bytesCopied);
  get_counter_dataPrefix_params.bytesAllocated =
      counter_data_image_prefix.size();
  get_counter_dataPrefix_params.pBuffer = counter_data_image_prefix.data();
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(
                                  &get_counter_dataPrefix_params));

  return true;
}

#else  // CUDA_VERSION < 11040

// copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool get_raw_metric_requests(
    NVPA_MetricsContext *metrics_context,
    std::vector<std::string> metric_names,
    std::vector<NVPA_RawMetricRequest> &raw_metric_requests,
    std::vector<std::string> &temp) {
  std::string req_name;
  bool isolated = true;
  bool keep_instances = true;

  for (auto &metric_name : metric_names) {
    parse_metric_name_string(metric_name, &req_name, &isolated,
                             &keep_instances);
    /* Bug in collection with collection of metrics without instances, keep it
     * to true*/
    keep_instances = true;
    NVPW_MetricsContext_GetMetricProperties_Begin_Params
        get_metric_properties_begin_params = {
            NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE};
    get_metric_properties_begin_params.pMetricsContext = metrics_context;
    get_metric_properties_begin_params.pMetricName = req_name.c_str();

    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_Begin(
                                    &get_metric_properties_begin_params));

    for (const char **pp_metric_dependencies =
             get_metric_properties_begin_params.ppRawMetricDependencies;
         *pp_metric_dependencies; ++pp_metric_dependencies) {
      temp.push_back(*pp_metric_dependencies);
    }
    NVPW_MetricsContext_GetMetricProperties_End_Params
        get_metric_properties_end_params = {
            NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE};
    get_metric_properties_end_params.pMetricsContext = metrics_context;
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_End(
                                    &get_metric_properties_end_params));
  }

  for (auto &raw_metric_name : temp) {
    NVPA_RawMetricRequest metric_request = {
        NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
    metric_request.pMetricName = raw_metric_name.c_str();
    metric_request.isolated = isolated;
    metric_request.keepInstances = keep_instances;
    raw_metric_requests.push_back(metric_request);
  }

  return true;
}

// copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool get_config_image(std::string chip_name,
                      std::vector<std::string> metric_names,
                      std::vector<uint8_t> &config_image,
                      const uint8_t *p_counter_availability_image) {
  NVPW_CUDA_MetricsContext_Create_Params metrics_context_create_params = {
      NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE};
  metrics_context_create_params.pChipName = chip_name.c_str();
  RETURN_IF_NVPW_ERROR(
      false, NVPW_CUDA_MetricsContext_Create(&metrics_context_create_params));

  NVPW_MetricsContext_Destroy_Params metrics_context_destroy_params = {
      NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE};
  metrics_context_destroy_params.pMetricsContext =
      metrics_context_create_params.pMetricsContext;
  SCOPE_EXIT([&]() {
    NVPW_MetricsContext_Destroy(
        (NVPW_MetricsContext_Destroy_Params *)&metrics_context_destroy_params);
  });

  std::vector<NVPA_RawMetricRequest> raw_metric_requests;
  std::vector<std::string> temp;
  get_raw_metric_requests(metrics_context_create_params.pMetricsContext,
                          metric_names, raw_metric_requests, temp);

  NVPA_RawMetricsConfigOptions metrics_config_options = {
      NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE};
  metrics_config_options.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
  metrics_config_options.pChipName = chip_name.c_str();
  NVPA_RawMetricsConfig *p_raw_metrics_config;
  RETURN_IF_NVPW_ERROR(false,
                       NVPA_RawMetricsConfig_Create(&metrics_config_options,
                                                    &p_raw_metrics_config));

  if (p_counter_availability_image) {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params
        set_counter_availability_params = {
            NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
    set_counter_availability_params.pRawMetricsConfig = p_raw_metrics_config;
    set_counter_availability_params.pCounterAvailabilityImage =
        p_counter_availability_image;
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_SetCounterAvailability(
                                    &set_counter_availability_params));
  }

  NVPW_RawMetricsConfig_Destroy_Params raw_metrics_config_destroy_params = {
      NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
  raw_metrics_config_destroy_params.pRawMetricsConfig = p_raw_metrics_config;
  SCOPE_EXIT([&]() {
    NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params
                                       *)&raw_metrics_config_destroy_params);
  });

  NVPW_RawMetricsConfig_BeginPassGroup_Params begin_pass_group_params = {
      NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
  begin_pass_group_params.pRawMetricsConfig = p_raw_metrics_config;
  RETURN_IF_NVPW_ERROR(
      false, NVPW_RawMetricsConfig_BeginPassGroup(&begin_pass_group_params));

  NVPW_RawMetricsConfig_AddMetrics_Params add_metrics_params = {
      NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
  add_metrics_params.pRawMetricsConfig = p_raw_metrics_config;
  add_metrics_params.pRawMetricRequests = &raw_metric_requests[0];
  add_metrics_params.numMetricRequests = raw_metric_requests.size();
  RETURN_IF_NVPW_ERROR(false,
                       NVPW_RawMetricsConfig_AddMetrics(&add_metrics_params));

  NVPW_RawMetricsConfig_EndPassGroup_Params end_pass_group_params = {
      NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
  end_pass_group_params.pRawMetricsConfig = p_raw_metrics_config;
  RETURN_IF_NVPW_ERROR(
      false, NVPW_RawMetricsConfig_EndPassGroup(&end_pass_group_params));

  NVPW_RawMetricsConfig_GenerateConfigImage_Params
      generate_config_image_params = {
          NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE};
  generate_config_image_params.pRawMetricsConfig = p_raw_metrics_config;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GenerateConfigImage(
                                  &generate_config_image_params));

  NVPW_RawMetricsConfig_GetConfigImage_Params get_config_image_params = {
      NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE};
  get_config_image_params.pRawMetricsConfig = p_raw_metrics_config;
  get_config_image_params.bytesAllocated = 0;
  get_config_image_params.pBuffer = NULL;
  RETURN_IF_NVPW_ERROR(
      false, NVPW_RawMetricsConfig_GetConfigImage(&get_config_image_params));

  config_image.resize(get_config_image_params.bytesCopied);

  get_config_image_params.bytesAllocated = config_image.size();
  get_config_image_params.pBuffer = &config_image[0];
  RETURN_IF_NVPW_ERROR(
      false, NVPW_RawMetricsConfig_GetConfigImage(&get_config_image_params));

  return true;
}

// copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool get_counter_data_prefix_image(
    std::string chip_name,
    std::vector<std::string> metric_names,
    std::vector<uint8_t> &counter_data_image_prefix,
    const uint8_t *p_counter_availability_image) {
  NVPW_CUDA_MetricsContext_Create_Params metrics_context_create_params = {
      NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE};
  metrics_context_create_params.pChipName = chip_name.c_str();
  RETURN_IF_NVPW_ERROR(
      false, NVPW_CUDA_MetricsContext_Create(&metrics_context_create_params));

  NVPW_MetricsContext_Destroy_Params metrics_context_destroy_params = {
      NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE};
  metrics_context_destroy_params.pMetricsContext =
      metrics_context_create_params.pMetricsContext;
  SCOPE_EXIT([&]() {
    NVPW_MetricsContext_Destroy(
        (NVPW_MetricsContext_Destroy_Params *)&metrics_context_destroy_params);
  });

  std::vector<NVPA_RawMetricRequest> raw_metric_requests;
  std::vector<std::string> temp;
  get_raw_metric_requests(metrics_context_create_params.pMetricsContext,
                          metric_names, raw_metric_requests, temp);

  NVPW_CounterDataBuilder_Create_Params counter_data_builder_create_params = {
      NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE};
  counter_data_builder_create_params.pChipName = chip_name.c_str();
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_Create(
                                  &counter_data_builder_create_params));

  NVPW_CounterDataBuilder_Destroy_Params counter_data_builder_destroy_params = {
      NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
  counter_data_builder_destroy_params.pCounterDataBuilder =
      counter_data_builder_create_params.pCounterDataBuilder;
  SCOPE_EXIT([&]() {
    NVPW_CounterDataBuilder_Destroy(
        (NVPW_CounterDataBuilder_Destroy_Params
             *)&counter_data_builder_destroy_params);
  });

  NVPW_CounterDataBuilder_AddMetrics_Params add_metrics_params = {
      NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
  add_metrics_params.pCounterDataBuilder =
      counter_data_builder_create_params.pCounterDataBuilder;
  add_metrics_params.pRawMetricRequests = &raw_metric_requests[0];
  add_metrics_params.numMetricRequests = raw_metric_requests.size();
  RETURN_IF_NVPW_ERROR(false,
                       NVPW_CounterDataBuilder_AddMetrics(&add_metrics_params));

  size_t counterDataPrefixSize = 0;
  NVPW_CounterDataBuilder_GetCounterDataPrefix_Params
      get_counter_data_prefix_params = {
          NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
  get_counter_data_prefix_params.pCounterDataBuilder =
      counter_data_builder_create_params.pCounterDataBuilder;
  get_counter_data_prefix_params.bytesAllocated = 0;
  get_counter_data_prefix_params.pBuffer = NULL;
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(
                                  &get_counter_data_prefix_params));

  counter_data_image_prefix.resize(get_counter_data_prefix_params.bytesCopied);

  get_counter_data_prefix_params.bytesAllocated =
      counter_data_image_prefix.size();
  get_counter_data_prefix_params.pBuffer = &counter_data_image_prefix[0];
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(
                                  &get_counter_data_prefix_params));

  return true;
}

#endif  // CUDA_VERSION

// copy from CUPTI/samples/autorange_profiling/simplecuda.cu
bool create_counter_data_image(
    uint32_t num_ranges,
    std::vector<uint8_t> &counter_data_image,
    std::vector<uint8_t> &counter_data_scratch_buffer,
    std::vector<uint8_t> &counter_data_image_prefix) {
  CUpti_Profiler_CounterDataImageOptions counter_data_image_options;
  counter_data_image_options.pCounterDataPrefix = &counter_data_image_prefix[0];
  counter_data_image_options.counterDataPrefixSize =
      counter_data_image_prefix.size();
  counter_data_image_options.maxNumRanges = num_ranges;
  counter_data_image_options.maxNumRangeTreeNodes = num_ranges;
  counter_data_image_options.maxRangeNameLength = 64;

  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculate_size_params = {
      CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

  calculate_size_params.pOptions = &counter_data_image_options;
  calculate_size_params.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  CUPTI_API_CALL(
      cuptiProfilerCounterDataImageCalculateSize(&calculate_size_params));

  CUpti_Profiler_CounterDataImage_Initialize_Params initialize_params = {
      CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  initialize_params.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initialize_params.pOptions = &counter_data_image_options;
  initialize_params.counterDataImageSize =
      calculate_size_params.counterDataImageSize;

  counter_data_image.resize(calculate_size_params.counterDataImageSize);
  initialize_params.pCounterDataImage = &counter_data_image[0];
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initialize_params));

  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
      scratch_buffer_size_params = {
          CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratch_buffer_size_params.counterDataImageSize =
      calculate_size_params.counterDataImageSize;
  scratch_buffer_size_params.pCounterDataImage =
      initialize_params.pCounterDataImage;
  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
      &scratch_buffer_size_params));

  counter_data_scratch_buffer.resize(
      scratch_buffer_size_params.counterDataScratchBufferSize);

  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
      init_scratch_buffer_params = {
          CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  init_scratch_buffer_params.counterDataImageSize =
      calculate_size_params.counterDataImageSize;

  init_scratch_buffer_params.pCounterDataImage =
      initialize_params.pCounterDataImage;
  init_scratch_buffer_params.counterDataScratchBufferSize =
      scratch_buffer_size_params.counterDataScratchBufferSize;
  init_scratch_buffer_params.pCounterDataScratchBuffer =
      &counter_data_scratch_buffer[0];

  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &init_scratch_buffer_params));

  return true;
}

CuptiToolkit::CuptiToolkit() {
  TI_TRACE("CuptiToolkit::CuptiToolkit() ");
  cupti_config_.metric_list.clear();
  uint metric_list_size =
      static_cast<uint>(CuptiMetricsDefault::CUPTI_METRIC_DEFAULT_TOTAL);
  for (uint idx = 0; idx < metric_list_size; idx++) {
    cupti_config_.metric_list.push_back(MetricListDefault[idx]);
  }
  set_status(true);
}

CuptiToolkit::~CuptiToolkit() {
  if (enabled_) {
    end_profiling();
    deinit_cupti();
  }
}

void CuptiToolkit::set_status(bool enable) {
  enabled_ = enable;
}

void CuptiToolkit::reset_metrics(const std::vector<std::string> &metrics) {
  cupti_config_.metric_list.clear();
  uint metric_list_size =
      static_cast<uint>(CuptiMetricsDefault::CUPTI_METRIC_DEFAULT_TOTAL);
  for (uint idx = 0; idx < metric_list_size; idx++) {
    cupti_config_.metric_list.push_back(MetricListDefault[idx]);
  }
  // user selected metrics
  for (auto metric : metrics)
    cupti_config_.metric_list.push_back(metric);
}

bool CuptiToolkit::init_cupti() {
  // copy from CUPTI/samples/autorange_profiling/simplecuda.cu
  CUpti_Profiler_Initialize_Params profiler_initialize_params = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerInitialize(&profiler_initialize_params));
  TI_TRACE("cuptiProfilerInitialized");

  int device_num = 0;  // TODO
  // CUDADriver::get_instance_without_context().init(0);
  CUpti_Device_GetChipName_Params get_chip_name_params = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  get_chip_name_params.deviceIndex = device_num;
  CUPTI_API_CALL(cuptiDeviceGetChipName(&get_chip_name_params));
  cupti_image_.chip_name = get_chip_name_params.pChipName;
  TI_TRACE("cuptiDeviceGetChipName : {}", cupti_image_.chip_name);

  CUpti_Profiler_GetCounterAvailability_Params get_counter_availability_params =
      {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  get_counter_availability_params.ctx =
      (CUcontext)(CUDAContext::get_instance().get_context());
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&get_counter_availability_params));

  TI_TRACE(
      "counterAvailabilityImageSize : {}",
      get_counter_availability_params.counterAvailabilityImageSize);  // 2192
  cupti_image_.counter_availability_image.clear();
  cupti_image_.counter_availability_image.resize(
      get_counter_availability_params.counterAvailabilityImageSize);
  get_counter_availability_params.pCounterAvailabilityImage =
      cupti_image_.counter_availability_image.data();
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&get_counter_availability_params));

#define NVPW_API_CALL(api_func_call)                                       \
  do {                                                                     \
    NVPA_Status status = api_func_call;                                    \
    if (status != NVPA_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
              __FILE__, __LINE__, #api_func_call, status);                 \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initialize_host_params = {
      NVPW_InitializeHost_Params_STRUCT_SIZE};
  NVPW_API_CALL(NVPW_InitializeHost(&initialize_host_params));
  TI_TRACE("NVPW_InitializeHost");

#undef NVPW_API_CALL

  bool state = 0;
  state = get_config_image(cupti_image_.chip_name, cupti_config_.metric_list,
                           cupti_image_.config_image,
                           cupti_image_.counter_availability_image.data());
  if (!state) {
    TI_ERROR("Failed to create config_image");
  }
  TI_TRACE("get_config_image");

  state = get_counter_data_prefix_image(
      cupti_image_.chip_name, cupti_config_.metric_list,
      cupti_image_.counter_data_image_prefix,
      cupti_image_.counter_availability_image.data());
  if (!state) {
    TI_ERROR("Failed to create counter_data_image_prefix");
  }
  TI_TRACE("get_counter_data_prefix_image");

  state = create_counter_data_image(cupti_config_.num_ranges,
                                    cupti_image_.counter_data_image,
                                    cupti_image_.counter_data_scratch_buffer,
                                    cupti_image_.counter_data_image_prefix);

  if (!state) {
    TI_ERROR("Failed to create counter_data_image");
  }
  TI_TRACE("create_counter_data_image");

  return true;
}

bool CuptiToolkit::begin_profiling() {
  // copy from CUPTI/samples/autorange_profiling/simplecuda.cu
  CUpti_Profiler_BeginSession_Params begin_session_params = {
      CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
  CUpti_Profiler_SetConfig_Params set_config_params = {
      CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  CUpti_Profiler_EnableProfiling_Params enable_profiling_params = {
      CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};

  begin_session_params.ctx = NULL;
  begin_session_params.counterDataImageSize =
      cupti_image_.counter_data_image.size();
  begin_session_params.pCounterDataImage =
      &(cupti_image_.counter_data_image[0]);
  begin_session_params.counterDataScratchBufferSize =
      cupti_image_.counter_data_scratch_buffer.size();
  begin_session_params.pCounterDataScratchBuffer =
      &(cupti_image_.counter_data_scratch_buffer[0]);
  begin_session_params.range = CUPTI_AutoRange;          // hardcode
  begin_session_params.replayMode = CUPTI_KernelReplay;  // hardcode
  begin_session_params.maxRangesPerPass = cupti_config_.num_ranges;
  begin_session_params.maxLaunchesPerPass = cupti_config_.num_ranges;

  CUPTI_API_CALL(cuptiProfilerBeginSession(&begin_session_params));

  set_config_params.pConfig = &(cupti_image_.config_image[0]);
  set_config_params.configSize = cupti_image_.config_image.size();

  if (begin_session_params.replayMode == CUPTI_KernelReplay) {
    set_config_params.passIndex = 0;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&set_config_params));
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enable_profiling_params));
  } else {
    TI_ERROR("begin_session_params.replayMode != CUPTI_KernelReplay");
  }
  return true;
}

bool CuptiToolkit::end_profiling() {
  CUpti_Profiler_DisableProfiling_Params disable_profiling_params = {
      CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disable_profiling_params));
  CUpti_Profiler_UnsetConfig_Params unset_config_params = {
      CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unset_config_params));
  CUpti_Profiler_EndSession_Params end_session_params = {
      CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerEndSession(&end_session_params));
  return true;
}

bool CuptiToolkit::deinit_cupti() {
  CUpti_Profiler_DeInitialize_Params profiler_deinitialize_params = {
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDeInitialize(&profiler_deinitialize_params));
  return true;
}

bool CuptiToolkit::update_record(
    uint32_t records_size_after_sync,
    std::vector<KernelProfileTracedRecord> &traced_records) {
  if (!cupti_image_.counter_data_image.size()) {
    TI_WARN("Counter Data Image is empty!");
    return false;
  }

  // copy from CUPTI/samples/autorange_profiling/simplecuda.cu
  NVPW_CUDA_MetricsContext_Create_Params metrics_context_create_params = {
      NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE};
  metrics_context_create_params.pChipName = cupti_image_.chip_name.c_str();
  RETURN_IF_NVPW_ERROR(
      false, NVPW_CUDA_MetricsContext_Create(&metrics_context_create_params));

  NVPW_MetricsContext_Destroy_Params metrics_context_destroy_params = {
      NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE};
  metrics_context_destroy_params.pMetricsContext =
      metrics_context_create_params.pMetricsContext;
  SCOPE_EXIT([&]() {
    NVPW_MetricsContext_Destroy(
        (NVPW_MetricsContext_Destroy_Params *)&metrics_context_destroy_params);
  });

  NVPW_CounterData_GetNumRanges_Params get_num_ranges_params = {
      NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE};
  get_num_ranges_params.pCounterDataImage =
      &(cupti_image_.counter_data_image[0]);
  RETURN_IF_NVPW_ERROR(false,
                       NVPW_CounterData_GetNumRanges(&get_num_ranges_params));

  std::vector<std::string> req_name;
  req_name.resize(cupti_config_.metric_list.size());
  bool isolated = true;
  bool keep_instances = true;
  std::vector<const char *> metric_name_ptrs;
  for (size_t metric_index = 0; metric_index < cupti_config_.metric_list.size();
       ++metric_index) {
    bool status = parse_metric_name_string(
        cupti_config_.metric_list[metric_index], &req_name[metric_index],
        &isolated, &keep_instances);
    if (!status) {
      TI_ERROR("parse_metric_name_string error !");
      return false;
    }
    metric_name_ptrs.push_back(req_name[metric_index].c_str());
  }

  TI_TRACE("get_num_ranges_params.numRanges = {}",
           get_num_ranges_params.numRanges);
  for (size_t range_index = 0; range_index < get_num_ranges_params.numRanges;
       ++range_index) {
    std::vector<const char *> description_ptrs;

    NVPW_Profiler_CounterData_GetRangeDescriptions_Params
        get_range_desc_params = {
            NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE};
    get_range_desc_params.pCounterDataImage =
        &(cupti_image_.counter_data_image[0]);
    get_range_desc_params.rangeIndex = range_index;
    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(
                                    &get_range_desc_params));

    description_ptrs.resize(get_range_desc_params.numDescriptions);
    get_range_desc_params.ppDescriptions = &description_ptrs[0];
    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(
                                    &get_range_desc_params));

    std::string range_name;
    for (size_t description_index = 0;
         description_index < get_range_desc_params.numDescriptions;
         ++description_index) {
      if (description_index) {
        range_name += "/";
      }
      range_name += description_ptrs[description_index];
    }

    const bool isolated = true;
    std::vector<double> gpu_values;
    gpu_values.resize(cupti_config_.metric_list.size());

    NVPW_MetricsContext_SetCounterData_Params set_counter_data_params = {
        NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE};
    set_counter_data_params.pMetricsContext =
        metrics_context_create_params.pMetricsContext;
    set_counter_data_params.pCounterDataImage =
        &(cupti_image_.counter_data_image[0]);
    set_counter_data_params.isolated = true;
    set_counter_data_params.rangeIndex = range_index;
    RETURN_IF_NVPW_ERROR(
        false, NVPW_MetricsContext_SetCounterData(&set_counter_data_params));

    NVPW_MetricsContext_EvaluateToGpuValues_Params eval_to_gpu_params = {
        NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE};
    eval_to_gpu_params.pMetricsContext =
        metrics_context_create_params.pMetricsContext;
    eval_to_gpu_params.numMetrics = metric_name_ptrs.size();
    eval_to_gpu_params.ppMetricNames = &metric_name_ptrs[0];
    eval_to_gpu_params.pMetricValues = &gpu_values[0];
    RETURN_IF_NVPW_ERROR(
        false, NVPW_MetricsContext_EvaluateToGpuValues(&eval_to_gpu_params));

    // default metric : kernel_elapsed_time_in_ms
    double kernel_elapsed_clk_nums = gpu_values[static_cast<uint>(
        CuptiMetricsDefault::CUPTI_METRIC_KERNEL_ELAPSED_CLK_NUMS)];
    double core_frequency_hzs = gpu_values[static_cast<uint>(
        CuptiMetricsDefault::CUPTI_METRIC_CORE_FREQUENCY_HZS)];
    traced_records[records_size_after_sync + range_index]
        .kernel_elapsed_time_in_ms =
        kernel_elapsed_clk_nums / core_frequency_hzs * 1000;  // from s to ms

    // user selected metrics
    uint user_metric_idx_begin =
        static_cast<uint>(CuptiMetricsDefault::CUPTI_METRIC_DEFAULT_TOTAL);
    uint metric_num = cupti_config_.metric_list.size();
    for (uint idx = user_metric_idx_begin; idx < metric_num; idx++) {
      traced_records[records_size_after_sync + range_index]
          .metric_values.push_back(gpu_values[idx]);
    }
  }
  return true;
}

// undef macros
#undef NV_ANONYMOUS_VARIABLE_DIRECT
#undef NV_ANONYMOUS_VARIABLE_INDIRECT
#undef CUPTI_API_CALL
#undef SCOPE_EXIT
#undef RETURN_IF_NVPW_ERROR

#else

CuptiToolkit::CuptiToolkit() {
  TI_NOT_IMPLEMENTED;
}
CuptiToolkit::~CuptiToolkit() {
  TI_NOT_IMPLEMENTED;
}
void CuptiToolkit::set_status(bool enable) {
  TI_NOT_IMPLEMENTED;
}
void CuptiToolkit::reset_metrics(const std::vector<std::string> &metrics) {
  TI_NOT_IMPLEMENTED;
}
bool CuptiToolkit::init_cupti() {
  TI_NOT_IMPLEMENTED;
}
bool CuptiToolkit::begin_profiling() {
  TI_NOT_IMPLEMENTED;
}
bool CuptiToolkit::end_profiling() {
  TI_NOT_IMPLEMENTED;
}
bool CuptiToolkit::deinit_cupti() {
  TI_NOT_IMPLEMENTED;
}
bool CuptiToolkit::update_record(
    uint32_t records_size_after_sync,
    std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}
#endif

TLANG_NAMESPACE_END
