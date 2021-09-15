#include "taichi/backends/cuda/cupti_toolkit.h"
#include "taichi/backends/cuda/cuda_context.h"

TLANG_NAMESPACE_BEGIN

bool check_device_capability() {
  void *device;
  int cc_major;
  CUDADriver::get_instance().device_get(&device, 0);
  CUDADriver::get_instance().device_get_attribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (cc_major < 7) {
    TI_WARN(
        "CUPTI profiler APIs unsupported on Device with compute capability < "
        "7.0 , fallback to default kernel profiler");
    return false;
  }
  return true;
}

bool check_cupti_privileges() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  CUpti_Profiler_Initialize_Params init_param = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUptiResult status = cuptiProfilerInitialize(&init_param);
  if (status == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) {
    TI_WARN(
        "function cuptiProfilerInitialize failed with error : "
        "CUPTI_ERROR_INSUFFICIENT_PRIVILEGES");
    TI_WARN("fallback to default kernel profiler : cuEvent");
    TI_WARN(
        "=================================================================");
    TI_WARN("Run your commands with `sudo` to get administrative privileges");
    TI_WARN("Add option: `nvidia NVreg_RestrictProfilingToAdminUsers=0` ");
    TI_WARN("to  /etc/modprobe.d/nvidia-kernel-common.conf");
    TI_WARN("then `reboot` should resolve the permision issue.");
    TI_WARN("( Probably needs to run `update-initramfs -u`  before `reboot` )");
    TI_WARN(
        "=================================================================");
    // TODO : doc and web
    return false;
  }
  // if there are other errors , CuptiToolkit::init_cupti() will send error
  // message
  return true;
#else
  return false;
#endif
}

#if defined(TI_WITH_CUDA_TOOLKIT)

CuptiToolkit::CuptiToolkit() {
  TI_TRACE("CuptiToolkit::CuptiToolkit() ");
  cupti_config_.metric_list.clear();
  for (uint32_t idx = 0; idx < CuptiMetricsDefault::CUPTI_METRIC_DEFAULT_TOTAL;
       idx++)
    cupti_config_.metric_list.push_back(MetricListDeafult[idx]);
}

CuptiToolkit::~CuptiToolkit() {
  end_profiling();
  deinit_cupti();
}

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
                      const uint8_t *p_pounter_availability_image) {
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

  if (p_pounter_availability_image) {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params
        set_counter_availability_params = {
            NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
    set_counter_availability_params.pRawMetricsConfig = p_raw_metrics_config;
    set_counter_availability_params.pCounterAvailabilityImage =
        p_pounter_availability_image;
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
    std::vector<uint8_t> &counter_data_image_prefix) {
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

  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initialize_host_params = {
      NVPW_InitializeHost_Params_STRUCT_SIZE};
  NVPW_API_CALL(NVPW_InitializeHost(&initialize_host_params));
  TI_TRACE("NVPW_InitializeHost");

  bool state = 0;
  state = get_config_image(cupti_image_.chip_name, cupti_config_.metric_list,
                           cupti_image_.config_image,
                           cupti_image_.counter_availability_image.data());
  if (!state) {
    TI_ERROR("Failed to create config_image");
  }
  TI_TRACE("get_config_image");

  state = get_counter_data_prefix_image(cupti_image_.chip_name,
                                        cupti_config_.metric_list,
                                        cupti_image_.counter_data_image_prefix);
  if (!state) {
    TI_ERROR("Failed to create counter_data_image_prefix");
  }
  TI_TRACE("get_counter_data_prefix_image");

  state = create_counter_data_image(cupti_config_.num_ranges,
                                    cupti_image_.counter_data_image,
                                    cupti_image_.counter_data_scratch_buffer,
                                    cupti_image_.counter_data_image_prefix);
  if (!state) {
    TI_ERROR("Failed to create counterDataImage");
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
  begin_session_params.range = cupti_config_.profiler_range;
  begin_session_params.replayMode = cupti_config_.profiler_replay_mode;
  begin_session_params.maxRangesPerPass = cupti_config_.num_ranges;
  begin_session_params.maxLaunchesPerPass = cupti_config_.num_ranges;

  CUPTI_API_CALL(cuptiProfilerBeginSession(&begin_session_params));

  set_config_params.pConfig = &(cupti_image_.config_image[0]);
  set_config_params.configSize = cupti_image_.config_image.size();

  if (cupti_config_.profiler_replay_mode == CUPTI_KernelReplay) {
    set_config_params.passIndex = 0;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&set_config_params));
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enable_profiling_params));
  } else {
    TI_ERROR("profiler_replay_mode != CUPTI_KernelReplay");
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

    traced_records[range_index].kernel_elapsed_time_in_ms =
        gpu_values[CUPTI_METRIC_KERNEL_ELAPSED_CLK_NUMS] /
        gpu_values[CUPTI_METRIC_CORE_FREQUENCY_HZS] * 1000;  // from s to ms
    // traced_records[range_index].memory_load_byets  =
    // gpu_values[CUPTI_METRIC_GLOBAL_LOAD_BYTES];
    // traced_records[range_index].memory_store_byets =
    // gpu_values[CUPTI_METRIC_GLOBAL_STORE_BYTES];
    // TODO add these metrics value to record(backend and frontend)
  }
  return true;
}

// undef macros defined in ./cupti_toolkit_functions.h
#undef CUPTI_API_CALL
#undef NVPW_API_CALL
#undef NV_ANONYMOUS_VARIABLE_DIRECT
#undef NV_ANONYMOUS_VARIABLE_INDIRECT
#undef SCOPE_EXIT
#undef RETURN_IF_NVPW_ERROR
#undef NVPW_STATUS_RESULT

#else

CuptiToolkit::CuptiToolkit() {
  TI_NOT_IMPLEMENTED;
}
CuptiToolkit::~CuptiToolkit() {
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
    std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}
#endif

TLANG_NAMESPACE_END
