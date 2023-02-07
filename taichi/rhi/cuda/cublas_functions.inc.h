PER_CUBLAS_FUNCTION(cubCreate, cublasCreate_v2, cublasHandle_t *);
PER_CUBLAS_FUNCTION(cubDestroy, cublasDestroy_v2, cublasHandle_t);
PER_CUBLAS_FUNCTION(cubGetVersion, cublasGetVersion_v2, cublasHandle_t, int *);
PER_CUBLAS_FUNCTION(cubSaxpy,
                    cublasSaxpy_v2,
                    cublasHandle_t,
                    int,
                    const float *,
                    const float *,
                    int,
                    float *,
                    int);
PER_CUBLAS_FUNCTION(cubSnrm2,
                    cublasSnrm2_v2,
                    cublasHandle_t,
                    int,
                    const float *,
                    int,
                    float *);
PER_CUBLAS_FUNCTION(cubSdot,
                    cublasSdot_v2,
                    cublasHandle_t,
                    int,
                    const float *,
                    int,
                    const float *,
                    int,
                    float *);
PER_CUBLAS_FUNCTION(cubSscal,
                    cublasSscal_v2,
                    cublasHandle_t,
                    int,
                    const float *,
                    float *,
                    int);
