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
PER_CUSPARSE_FUNCTION(cpSetMatType, cusparseSetMatType, cusparseMatDescr_t, cusparseMatrixType_t);
PER_CUSPARSE_FUNCTION(cpSetMatIndexBase, cusparseSetMatIndexBase, cusparseMatDescr_t, cusparseIndexBase_t);
PER_CUSPARSE_FUNCTION(cpDestroySpMat, cusparseDestroySpMat, cusparseSpMatDescr_t);

// cusparse dense vector description
PER_CUSPARSE_FUNCTION(cpCreateDnVec, cusparseCreateDnVec, cusparseDnVecDescr_t*, int, void*, cudaDataType);
PER_CUSPARSE_FUNCTION(cpDestroyDnVec, cusparseDestroyDnVec, cusparseDnVecDescr_t);

// cusparse sparse matrix-vector multiplication
PER_CUSPARSE_FUNCTION(cpSpMV_bufferSize, cusparseSpMV_bufferSize, cusparseHandle_t, cusparseOperation_t, const void*,cusparseSpMatDescr_t, cusparseDnVecDescr_t,const void*, cusparseDnVecDescr_t,cudaDataType, cusparseSpMVAlg_t, size_t*);
PER_CUSPARSE_FUNCTION(cpSpMV, cusparseSpMV, cusparseHandle_t, cusparseOperation_t, const void*,cusparseSpMatDescr_t, cusparseDnVecDescr_t,const void*, cusparseDnVecDescr_t,cudaDataType, cusparseSpMVAlg_t, void*);
