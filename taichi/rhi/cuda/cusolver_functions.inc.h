// clang-format off

// cusolver functions
PER_CUSOLVER_FUNCTION(csGetProperty, cusolverGetProperty, libraryPropertyType, int* );


// cusolver functions for solve A*x = b
PER_CUSOLVER_FUNCTION(csSpCreate, cusolverSpCreate, cusolverSpHandle_t * );
PER_CUSOLVER_FUNCTION(csSpDestory, cusolverSpDestroy, cusolverSpHandle_t );
PER_CUSOLVER_FUNCTION(csSpXcsrsymrcmHost, cusolverSpXcsrsymrcmHost, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,const int *,const int *,int *);
PER_CUSOLVER_FUNCTION(csSpXcsrperm_bufferSizeHost, cusolverSpXcsrperm_bufferSizeHost, cusolverSpHandle_t ,int ,int ,int ,const cusparseMatDescr_t ,int *,int *,const int *,const int *,size_t *);
PER_CUSOLVER_FUNCTION(cusolverSpXcsrpermHost, cusolverSpXcsrpermHost, cusolverSpHandle_t ,int ,int ,int ,const cusparseMatDescr_t ,int *,int *,const int *,const int *,int *,void *);
PER_CUSOLVER_FUNCTION(cusolverSpScsrlsvchol, cusolverSpScsrlsvchol, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,const float *,const int *,const int *,const float *,float ,int ,float *,int *);
