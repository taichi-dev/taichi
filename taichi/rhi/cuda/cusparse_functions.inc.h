// clang-format off

// cusparse setup
PER_CUSPARSE_FUNCTION(cpCreate, cusparseCreate, cusparseHandle_t *);
PER_CUSPARSE_FUNCTION(cpDestroy, cusparseDestroy, cusparseHandle_t);

// cusparse sparse matrix description
PER_CUSPARSE_FUNCTION(cpCreateCoo, cusparseCreateCoo, cusparseSpMatDescr_t*, int, int, int,void*, void*, void*,cusparseIndexType_t, cusparseIndexBase_t,cudaDataType );
PER_CUSPARSE_FUNCTION(cpCreateCsr, cusparseCreateCsr, cusparseSpMatDescr_t*, int, int, int,void*, void*, void*,cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t,cudaDataType );
PER_CUSPARSE_FUNCTION(cpCoo2Csr, cusparseXcoo2csr, cusparseHandle_t ,const void*, int, int,void*, cusparseIndexBase_t );
PER_CUSPARSE_FUNCTION(cpCsrGet, cusparseCsrGet, cusparseSpMatDescr_t ,size_t *,size_t *,size_t *,void**,void**,void**,cusparseIndexType_t* ,cusparseIndexType_t* ,cusparseIndexBase_t* ,cudaDataType*);
PER_CUSPARSE_FUNCTION(cpCreateMatDescr, cusparseCreateMatDescr, cusparseMatDescr_t*);
PER_CUSPARSE_FUNCTION(cpDestroyMatDescr, cusparseDestroyMatDescr, cusparseMatDescr_t);
PER_CUSPARSE_FUNCTION(cpSetMatType, cusparseSetMatType, cusparseMatDescr_t, cusparseMatrixType_t);
PER_CUSPARSE_FUNCTION(cpSetMatIndexBase, cusparseSetMatIndexBase, cusparseMatDescr_t, cusparseIndexBase_t);
PER_CUSPARSE_FUNCTION(cpDestroySpMat, cusparseDestroySpMat, cusparseSpMatDescr_t);
PER_CUSPARSE_FUNCTION(cpCreateSpVec, cusparseCreateSpVec, cusparseSpVecDescr_t* ,int ,int,void*,void*,cusparseIndexType_t,cusparseIndexBase_t,cudaDataType);
PER_CUSPARSE_FUNCTION(cpDestroySpVec, cusparseDestroySpVec, cusparseSpVecDescr_t);
PER_CUSPARSE_FUNCTION(cpCreateIdentityPermutation, cusparseCreateIdentityPermutation, cusparseHandle_t, int, void*);
PER_CUSPARSE_FUNCTION(cpXcoosort_bufferSizeExt, cusparseXcoosort_bufferSizeExt, cusparseHandle_t,int ,int,int, void* ,void* ,void*);
PER_CUSPARSE_FUNCTION(cpXcoosortByRow, cusparseXcoosortByRow, cusparseHandle_t,int,int,int,void* ,void* ,void* ,void*);
PER_CUSPARSE_FUNCTION(cpGather, cusparseGather, cusparseHandle_t, cusparseDnVecDescr_t, cusparseSpVecDescr_t);
PER_CUSPARSE_FUNCTION(cpScatter, cusparseScatter, cusparseHandle_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t);
PER_CUSPARSE_FUNCTION(cpSetPointerMode, cusparseSetPointerMode, cusparseHandle_t, cusparsePointerMode_t);
PER_CUSPARSE_FUNCTION(cpCsrSetPointers, cusparseCsrSetPointers, cusparseSpMatDescr_t,void*, void*, void*);

// cusparse dense vector description
PER_CUSPARSE_FUNCTION(cpCreateDnVec, cusparseCreateDnVec, cusparseDnVecDescr_t*, int, void*, cudaDataType);
PER_CUSPARSE_FUNCTION(cpDestroyDnVec, cusparseDestroyDnVec, cusparseDnVecDescr_t);

// cusparse sparse matrix-vector multiplication
PER_CUSPARSE_FUNCTION(cpSpMV_bufferSize, cusparseSpMV_bufferSize, cusparseHandle_t, cusparseOperation_t, const void*,cusparseSpMatDescr_t, cusparseDnVecDescr_t,const void*, cusparseDnVecDescr_t,cudaDataType, cusparseSpMVAlg_t, size_t*);
PER_CUSPARSE_FUNCTION(cpSpMV, cusparseSpMV, cusparseHandle_t, cusparseOperation_t, const void*,cusparseSpMatDescr_t, cusparseDnVecDescr_t,const void*, cusparseDnVecDescr_t,cudaDataType, cusparseSpMVAlg_t, void*);


// cusparse sparse matrix-matrix operation
PER_CUSPARSE_FUNCTION(cpScsrgeam2_bufferSizeExt, cusparseScsrgeam2_bufferSizeExt, cusparseHandle_t  ,int,int,void*,const cusparseMatDescr_t ,int, void*, void*, void*,void*,const cusparseMatDescr_t ,int, void*,void*,void*,const cusparseMatDescr_t ,void*,void*,void*,void*);
PER_CUSPARSE_FUNCTION(cpGetSize, cusparseSpMatGetSize, cusparseSpMatDescr_t, size_t* , size_t* , size_t*);
PER_CUSPARSE_FUNCTION(cpXcsrgeam2Nnz, cusparseXcsrgeam2Nnz, cusparseHandle_t,int,int,const cusparseMatDescr_t ,int,void*,void*,const cusparseMatDescr_t ,int,void*,void*,const cusparseMatDescr_t ,void*,void*,void*);
PER_CUSPARSE_FUNCTION(cpScsrgeam2, cusparseScsrgeam2, cusparseHandle_t,int,int,void*,const cusparseMatDescr_t ,int, void*,void*,void*,void*,const cusparseMatDescr_t ,int,void*,void*,void*,const cusparseMatDescr_t ,void*,void*,void*,void*);
PER_CUSPARSE_FUNCTION(cpSpGEMM_workEstimation, cusparseSpGEMM_workEstimation, cusparseHandle_t,cusparseOperation_t,cusparseOperation_t,const void*,cusparseSpMatDescr_t,cusparseSpMatDescr_t,const void*,cusparseSpMatDescr_t,cudaDataType,cusparseSpGEMMAlg_t,cusparseSpGEMMDescr_t,size_t*,void*);
PER_CUSPARSE_FUNCTION(cpSpGEMM_compute, cusparseSpGEMM_compute, cusparseHandle_t,cusparseOperation_t,cusparseOperation_t,const void*,cusparseSpMatDescr_t,cusparseSpMatDescr_t,const void*,cusparseSpMatDescr_t,cudaDataType,cusparseSpGEMMAlg_t,cusparseSpGEMMDescr_t,size_t*,void*);
PER_CUSPARSE_FUNCTION(cpSpGEMM_copy, cusparseSpGEMM_copy, cusparseHandle_t,cusparseOperation_t,cusparseOperation_t,const void*,cusparseSpMatDescr_t,cusparseSpMatDescr_t,const void*,cusparseSpMatDescr_t,cudaDataType,cusparseSpGEMMAlg_t,cusparseSpGEMMDescr_t);
PER_CUSPARSE_FUNCTION(cpCreateSpGEMM, cusparseSpGEMM_createDescr, cusparseSpGEMMDescr_t* );
PER_CUSPARSE_FUNCTION(cpDestroySpGEMM, cusparseSpGEMM_destroyDescr, cusparseSpGEMMDescr_t );

// cusparse sparse matrix convertions
PER_CUSPARSE_FUNCTION(cpCsr2cscEx2_bufferSize, cusparseCsr2cscEx2_bufferSize, cusparseHandle_t, int, int, int, const void*, const int*, const int*, void*, int*, int*, cudaDataType, cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, size_t*);
PER_CUSPARSE_FUNCTION(cpCsr2cscEx2, cusparseCsr2cscEx2, cusparseHandle_t, int, int, int, const void*, const int*, const int*, void*, int*, int*, cudaDataType, cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, void*);
