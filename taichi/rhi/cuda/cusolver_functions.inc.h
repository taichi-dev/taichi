// clang-format off

// cusolver functions
PER_CUSOLVER_FUNCTION(csGetProperty, cusolverGetProperty, libraryPropertyType, void * );


// cusolver functions for solve A*x = b
PER_CUSOLVER_FUNCTION(csSpCreate, cusolverSpCreate, cusolverSpHandle_t * );
PER_CUSOLVER_FUNCTION(csSpDestory, cusolverSpDestroy, cusolverSpHandle_t );
PER_CUSOLVER_FUNCTION(csSpXcsrissymHost, cusolverSpXcsrissymHost, cusolverSpHandle_t,int ,int ,const cusparseMatDescr_t ,const void *,const void *,const void *,void *);
PER_CUSOLVER_FUNCTION(csSpXcsrsymrcmHost, cusolverSpXcsrsymrcmHost, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,const void *,const void *,void *);
PER_CUSOLVER_FUNCTION(csSpXcsrsymamdHost, cusolverSpXcsrsymamdHost, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,const void *,const void *,void *);
PER_CUSOLVER_FUNCTION(csSpXcsrperm_bufferSizeHost, cusolverSpXcsrperm_bufferSizeHost, cusolverSpHandle_t ,int ,int ,int ,const cusparseMatDescr_t ,void *,void *,const void *,const void *,size_t *);
PER_CUSOLVER_FUNCTION(csSpXcsrpermHost, cusolverSpXcsrpermHost, cusolverSpHandle_t ,int ,int ,int ,const cusparseMatDescr_t ,void *,void *,const void *,const void *,void *,void *);
PER_CUSOLVER_FUNCTION(csSpScsrlsvcholHost, cusolverSpScsrlsvcholHost, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,const void *,const void *,const void *,const void *,float ,int ,void *,void *);

// cusolver preview API for cholesky
PER_CUSOLVER_FUNCTION(csSpCreateCsrcholInfo, cusolverSpCreateCsrcholInfo, csrcholInfo_t*);
PER_CUSOLVER_FUNCTION(csSpDestroyCsrcholInfo, cusolverSpDestroyCsrcholInfo, csrcholInfo_t);
PER_CUSOLVER_FUNCTION(csSpXcsrcholAnalysis, cusolverSpXcsrcholAnalysis,  cusolverSpHandle_t,int ,int ,const cusparseMatDescr_t , void *, void *,csrcholInfo_t );
PER_CUSOLVER_FUNCTION(csSpScsrcholBufferInfo, cusolverSpScsrcholBufferInfo,  cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,void *,void *,void *,csrcholInfo_t ,size_t *,size_t *);
PER_CUSOLVER_FUNCTION(csSpScsrcholFactor, cusolverSpScsrcholFactor, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,void *,void *,void *,csrcholInfo_t ,void *);
PER_CUSOLVER_FUNCTION(csSpScsrcholZeroPivot, cusolverSpScsrcholZeroPivot, cusolverSpHandle_t, csrcholInfo_t ,float ,void *);
PER_CUSOLVER_FUNCTION(csSpScsrcholSolve, cusolverSpScsrcholSolve,  cusolverSpHandle_t ,int ,void *,void *,csrcholInfo_t ,void *);

// cusolver preview API for LU
PER_CUSOLVER_FUNCTION(csSpCreateCsrluInfoHost, cusolverSpCreateCsrluInfoHost, csrluInfoHost_t*);
PER_CUSOLVER_FUNCTION(csSpDestroyCsrluInfoHost, cusolverSpDestroyCsrluInfoHost, csrluInfoHost_t);
PER_CUSOLVER_FUNCTION(csSpXcsrluAnalysisHost, cusolverSpXcsrluAnalysisHost,  cusolverSpHandle_t,int ,int ,const cusparseMatDescr_t , void *, void *,csrluInfoHost_t );
PER_CUSOLVER_FUNCTION(csSpScsrluBufferInfoHost, cusolverSpScsrluBufferInfoHost,  cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,void *,void *,void *,csrluInfoHost_t ,size_t *,size_t *);
PER_CUSOLVER_FUNCTION(csSpScsrluFactorHost, cusolverSpScsrluFactorHost, cusolverSpHandle_t ,int ,int ,const cusparseMatDescr_t ,void *,void *,void *,csrluInfoHost_t,float, void *);
PER_CUSOLVER_FUNCTION(csSpScsrluZeroPivotHost, cusolverSpScsrluZeroPivotHost, cusolverSpHandle_t, csrluInfoHost_t ,float ,void *);
PER_CUSOLVER_FUNCTION(csSpScsrluSolveHost, cusolverSpScsrluSolveHost,  cusolverSpHandle_t ,int ,void *,void *,csrluInfoHost_t ,void *);
