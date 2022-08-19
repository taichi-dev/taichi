// clang-format off

// cusolver functions
PER_CUSOLVER_FUNCTION(csGetProperty, cusolverGetProperty, libraryPropertyType, void * );


// cusolver functions for solve A*x = b
PER_CUSOLVER_FUNCTION(csSpCreate, cusolverSpCreate, cusolverSpHandle_t * );
PER_CUSOLVER_FUNCTION(csSpDestory, cusolverSpDestroy, cusolverSpHandle_t );
PER_CUSOLVER_FUNCTION(csSpXcsrissymHost, cusolverSpXcsrissymHost, cusolverSpHandle_t,int ,int ,const cusparseMatDescr_t ,const void *,const void *,const void *,void *);
PER_CUSOLVER_FUNCTION(csSpXcsrsymrcmHost, cusolverSpXcsrsymrcmHost, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,const void *,const void *,void *);
PER_CUSOLVER_FUNCTION(csSpXcsrperm_bufferSizeHost, cusolverSpXcsrperm_bufferSizeHost, cusolverSpHandle_t ,int ,int ,int ,const cusparseMatDescr_t ,void *,void *,const void *,const void *,size_t *);
PER_CUSOLVER_FUNCTION(csSpXcsrpermHost, cusolverSpXcsrpermHost, cusolverSpHandle_t ,int ,int ,int ,const cusparseMatDescr_t ,void *,void *,const void *,const void *,void *,void *);
PER_CUSOLVER_FUNCTION(csSpScsrlsvcholHost, cusolverSpScsrlsvcholHost, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,const void *,const void *,const void *,const void *,float ,int ,void *,void *);
