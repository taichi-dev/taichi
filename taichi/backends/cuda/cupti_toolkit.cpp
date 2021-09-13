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
  CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUptiResult status = cuptiProfilerInitialize(&profilerInitializeParams);
  if (status == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES){
    TI_WARN("function cuptiProfilerInitialize failed with error : CUPTI_ERROR_INSUFFICIENT_PRIVILEGES");
    TI_WARN("fallback to default kernel profiler : cuEvent");
    TI_WARN("=================================================================");
    TI_WARN("Run your commands with `sudo` to get administrative privileges");
    TI_WARN("Add option: `nvidia NVreg_RestrictProfilingToAdminUsers=0` ");
    TI_WARN("to  /etc/modprobe.d/nvidia-kernel-common.conf");
    TI_WARN("then `reboot` should resolve the permision issue.");
    TI_WARN("( Probably needs to run `update-initramfs -u`  before `reboot` )");
    TI_WARN("=================================================================");
    //TODO add [doc] and [web link]
    return false;
  }
  //if there are other errors , CuptiToolkit::init_cupti() will send error message 
  return true; 
#else
  return false;
#endif
}


#if defined(TI_WITH_CUDA_TOOLKIT)

CuptiToolkit::CuptiToolkit() {
  TI_TRACE("CuptiToolkit::CuptiToolkit() ");
  cupti_config_.metric_list.clear();
  for(uint32_t idx=0; idx< CuptiMetricsDefault::CUPTI_METRIC_DEFAULT_TOTAL; idx++ )
    cupti_config_.metric_list.push_back(MetricListDeafult[idx]);
}

CuptiToolkit::~CuptiToolkit() {
  end_profiling();
  deinit_cupti();
}

//copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool GetRawMetricRequests(NVPA_MetricsContext* pMetricsContext,
                          std::vector<std::string> metricNames,
                          std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
                          std::vector<std::string>& temp) {
    std::string reqName;
    bool isolated = true;
    bool keepInstances = true;

    for (auto& metricName : metricNames)
    {
        ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
        /* Bug in collection with collection of metrics without instances, keep it to true*/
        keepInstances = true;
        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
        getMetricPropertiesBeginParams.pMetricsContext = pMetricsContext;
        getMetricPropertiesBeginParams.pMetricName = reqName.c_str();

        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams));

        for (const char** ppMetricDependencies = getMetricPropertiesBeginParams.ppRawMetricDependencies; *ppMetricDependencies; ++ppMetricDependencies)
        {
            temp.push_back(*ppMetricDependencies);
        }
        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
        getMetricPropertiesEndParams.pMetricsContext = pMetricsContext;
        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_End(&getMetricPropertiesEndParams));
    }

    for (auto& rawMetricName : temp)
    {
        NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
        metricRequest.pMetricName = rawMetricName.c_str();
        metricRequest.isolated = isolated;
        metricRequest.keepInstances = keepInstances;
        rawMetricRequests.push_back(metricRequest);
    }

    return true;
}

//copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool GetConfigImage(std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& configImage, const uint8_t* pCounterAvailabilityImage) 
{
  NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
  metricsContextCreateParams.pChipName = chipName.c_str();
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

  NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
  metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
  SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });
  
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  std::vector<std::string> temp;
  GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);

  NVPA_RawMetricsConfigOptions metricsConfigOptions = { NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE };
  metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
  metricsConfigOptions.pChipName = chipName.c_str();
  NVPA_RawMetricsConfig* pRawMetricsConfig;
  RETURN_IF_NVPW_ERROR(false, NVPA_RawMetricsConfig_Create(&metricsConfigOptions, &pRawMetricsConfig));

  if(pCounterAvailabilityImage)
  {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
    setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
    setCounterAvailabilityParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_SetCounterAvailability(&setCounterAvailabilityParams));
  }

  NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
  rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
  SCOPE_EXIT([&]() { NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams); });

  NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
  beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

  NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
  addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
  addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

  NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
  endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

  NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
  generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

  NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
  getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
  getConfigImageParams.bytesAllocated = 0;
  getConfigImageParams.pBuffer = NULL;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

  configImage.resize(getConfigImageParams.bytesCopied);

  getConfigImageParams.bytesAllocated = configImage.size();
  getConfigImageParams.pBuffer = &configImage[0];
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

  return true;
}

//copy from : CUPTI/samples/extensions/src/profilerhost_util/Metric.cpp
bool GetCounterDataPrefixImage(std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& counterDataImagePrefix) 
{
  NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
  metricsContextCreateParams.pChipName = chipName.c_str();
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

  NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
  metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
  SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  std::vector<std::string> temp;
  GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);

  NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE };
  counterDataBuilderCreateParams.pChipName = chipName.c_str();
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

  NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
  counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
  SCOPE_EXIT([&]() { NVPW_CounterDataBuilder_Destroy((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams); });

  NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
  addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
  addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

  size_t counterDataPrefixSize = 0;
  NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
  getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
  getCounterDataPrefixParams.bytesAllocated = 0;
  getCounterDataPrefixParams.pBuffer = NULL;
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

  counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);

  getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
  getCounterDataPrefixParams.pBuffer = &counterDataImagePrefix[0];
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

  return true;
}

//copy from CUPTI/samples/autorange_profiling/simplecuda.cu
bool CreateCounterDataImage(
  uint32_t numRanges,
  std::vector<uint8_t>& counterDataImage,
  std::vector<uint8_t>& counterDataScratchBuffer,
  std::vector<uint8_t>& counterDataImagePrefix)
{

  CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
  counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
  counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
  counterDataImageOptions.maxNumRanges = numRanges;
  counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
  counterDataImageOptions.maxRangeNameLength = 64;

  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

  calculateSizeParams.pOptions = &counterDataImageOptions;
  calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

  CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initializeParams.pOptions = &counterDataImageOptions;
  initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

  counterDataImage.resize(calculateSizeParams.counterDataImageSize);
  initializeParams.pCounterDataImage = &counterDataImage[0];
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
  scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

  counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

  initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
  initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
  initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];

  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

  return true;
}


bool CuptiToolkit::init_cupti(){

  //copy from CUPTI/samples/autorange_profiling/simplecuda.cu
  CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
  TI_TRACE("cuptiProfilerInitialized");

  int deviceNum = 0; //TODO
  // CUDADriver::get_instance_without_context().init(0);
  CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
  getChipNameParams.deviceIndex = deviceNum;
  CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
  cupti_image_.chip_name = getChipNameParams.pChipName;
  TI_TRACE("cuptiDeviceGetChipName : {}",cupti_image_.chip_name);

  CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  getCounterAvailabilityParams.ctx = (CUcontext)(CUDAContext::get_instance().get_context());
  CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));
  
  TI_TRACE("counterAvailabilityImageSize : {}",getCounterAvailabilityParams.counterAvailabilityImageSize); //2192
  cupti_image_.counter_availability_image.clear();
  cupti_image_.counter_availability_image.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
  getCounterAvailabilityParams.pCounterAvailabilityImage = cupti_image_.counter_availability_image.data();
  CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
  NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));
  TI_TRACE("NVPW_InitializeHost");

  bool state = 0;
  state = GetConfigImage(cupti_image_.chip_name, cupti_config_.metric_list, cupti_image_.config_image, cupti_image_.counter_availability_image.data());
  if(!state){ TI_ERROR("Failed to create configImage"); }
  TI_TRACE("GetConfigImage");
  
  state = GetCounterDataPrefixImage(cupti_image_.chip_name, cupti_config_.metric_list, cupti_image_.counter_data_image_prefix);
  if(!state){ TI_ERROR("Failed to create counterDataImagePrefix"); }
  TI_TRACE("GetCounterDataPrefixImage");

  state = CreateCounterDataImage(cupti_config_.num_ranges, cupti_image_.counter_data_image, cupti_image_.counter_data_scratch_buffer, cupti_image_.counter_data_image_prefix);
  if(!state){ TI_ERROR("Failed to create counterDataImage");}
  TI_TRACE("CreateCounterDataImage");

  return true;
}

bool CuptiToolkit::begin_profiling(){

  //copy from CUPTI/samples/autorange_profiling/simplecuda.cu
  CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
  CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};

  beginSessionParams.ctx = NULL;
  beginSessionParams.counterDataImageSize = cupti_image_.counter_data_image.size();
  beginSessionParams.pCounterDataImage = &(cupti_image_.counter_data_image[0]);
  beginSessionParams.counterDataScratchBufferSize = cupti_image_.counter_data_scratch_buffer.size();
  beginSessionParams.pCounterDataScratchBuffer = &(cupti_image_.counter_data_scratch_buffer[0]);
  beginSessionParams.range = cupti_config_.profiler_range;
  beginSessionParams.replayMode = cupti_config_.profiler_replay_mode;
  beginSessionParams.maxRangesPerPass = cupti_config_.num_ranges;
  beginSessionParams.maxLaunchesPerPass = cupti_config_.num_ranges;

  CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

  setConfigParams.pConfig = &(cupti_image_.config_image[0]);
  setConfigParams.configSize = cupti_image_.config_image.size();

  if(cupti_config_.profiler_replay_mode == CUPTI_KernelReplay){
      setConfigParams.passIndex = 0;
      CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
      CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
  }
  else{
    TI_ERROR("profiler_replay_mode != CUPTI_KernelReplay");
  }
  return true;
}


bool CuptiToolkit::end_profiling(){
  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
  CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
  return true;
}

bool CuptiToolkit::deinit_cupti(){
  CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
  return true;
}


bool CuptiToolkit::update_record(std::vector<KernelProfileTracedRecord> &traced_records) {
  if (!cupti_image_.counter_data_image.size()) {
    TI_WARN("Counter Data Image is empty!");
    return false;
  }

  //copy from CUPTI/samples/autorange_profiling/simplecuda.cu
  NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
  metricsContextCreateParams.pChipName = cupti_image_.chip_name.c_str();
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

  NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
  metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
  SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

  NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
  getNumRangesParams.pCounterDataImage = &(cupti_image_.counter_data_image[0]);
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

  std::vector<std::string> reqName;
  reqName.resize(cupti_config_.metric_list.size());
  bool isolated = true;
  bool keepInstances = true;
  std::vector<const char*> metricNamePtrs;
  for (size_t metricIndex = 0; metricIndex < cupti_config_.metric_list.size(); ++metricIndex) {
    bool status = ParseMetricNameString(cupti_config_.metric_list[metricIndex], &reqName[metricIndex], &isolated, &keepInstances);\
    if (!status) {
      TI_ERROR("ParseMetricNameString error !");
      return false;
    }
    metricNamePtrs.push_back(reqName[metricIndex].c_str());
  }

  TI_TRACE("getNumRangesParams.numRanges = {}",getNumRangesParams.numRanges);
  for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex) {
    std::vector<const char*> descriptionPtrs;

    NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
    getRangeDescParams.pCounterDataImage = &(cupti_image_.counter_data_image[0]);
    getRangeDescParams.rangeIndex = rangeIndex;
    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

    descriptionPtrs.resize(getRangeDescParams.numDescriptions);
    getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

    std::string rangeName;
    for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
    {
      if (descriptionIndex)
      {
          rangeName += "/";
      }
      rangeName += descriptionPtrs[descriptionIndex];
    }

    const bool isolated = true;
    std::vector<double> gpuValues;
    gpuValues.resize(cupti_config_.metric_list.size());

    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
    setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    setCounterDataParams.pCounterDataImage = &(cupti_image_.counter_data_image[0]);
    setCounterDataParams.isolated = true;
    setCounterDataParams.rangeIndex = rangeIndex;
    RETURN_IF_NVPW_ERROR(false,NVPW_MetricsContext_SetCounterData(&setCounterDataParams));

    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
    evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    evalToGpuParams.numMetrics = metricNamePtrs.size();
    evalToGpuParams.ppMetricNames = &metricNamePtrs[0];
    evalToGpuParams.pMetricValues = &gpuValues[0];
    RETURN_IF_NVPW_ERROR(false,NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams));

    traced_records[rangeIndex].kernel_elapsed_time_in_ms = gpuValues[CUPTI_METRIC_KERNEL_ELAPSED_CLK_NUMS]
                                                           / gpuValues[CUPTI_METRIC_CORE_FREQUENCY_HZS]
                                                           * 1000 ; //from s to ms
    // traced_records[rangeIndex].memory_load_byets  = gpuValues[CUPTI_METRIC_GLOBAL_LOAD_BYTES];
    // traced_records[rangeIndex].memory_store_byets = gpuValues[CUPTI_METRIC_GLOBAL_STORE_BYTES];
    // TODO add these metrics value to record(backend and frontend)
  }
  return true;
}

#else

CuptiToolkit::CuptiToolkit(){
  TI_NOT_IMPLEMENTED;
}
CuptiToolkit::~CuptiToolkit(){
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
bool CuptiToolkit::update_record(std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}
#endif

TLANG_NAMESPACE_END