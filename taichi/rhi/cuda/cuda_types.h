#pragma once

#if defined(TI_WITH_CUDA_TOOLKIT)

#include <cuda.h>
#include <cusparse.h>

#else

using CUexternalMemory = void *;
using CUexternalSemaphore = void *;
using CUsurfObject = uint64_t;
using CUstream = void *;
using CUdeviceptr = void *;
using CUmipmappedArray = void *;
using CUarray = void *;

// copied from <cuda.h>

/**
 * Resource types
 */
typedef enum CUresourcetype_enum {
  CU_RESOURCE_TYPE_ARRAY = 0x00,           /**< Array resource */
  CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
  CU_RESOURCE_TYPE_LINEAR = 0x02,          /**< Linear resource */
  CU_RESOURCE_TYPE_PITCH2D = 0x03          /**< Pitch 2D resource */
} CUresourcetype;

/**
 * Array formats
 */
typedef enum CUarray_format_enum {
  CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,  /**< Unsigned 8-bit integers */
  CU_AD_FORMAT_UNSIGNED_INT16 = 0x02, /**< Unsigned 16-bit integers */
  CU_AD_FORMAT_UNSIGNED_INT32 = 0x03, /**< Unsigned 32-bit integers */
  CU_AD_FORMAT_SIGNED_INT8 = 0x08,    /**< Signed 8-bit integers */
  CU_AD_FORMAT_SIGNED_INT16 = 0x09,   /**< Signed 16-bit integers */
  CU_AD_FORMAT_SIGNED_INT32 = 0x0a,   /**< Signed 32-bit integers */
  CU_AD_FORMAT_HALF = 0x10,           /**< 16-bit floating point */
  CU_AD_FORMAT_FLOAT = 0x20           /**< 32-bit floating point */
} CUarray_format;

typedef enum CUfunction_attribute_enum {
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
  CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;

/**
 * 3D array descriptor
 */
typedef struct CUDA_ARRAY3D_DESCRIPTOR_st {
  size_t Width;  /**< Width of 3D array */
  size_t Height; /**< Height of 3D array */
  size_t Depth;  /**< Depth of 3D array */

  CUarray_format Format;    /**< Array format */
  unsigned int NumChannels; /**< Channels per array element */
  unsigned int Flags;       /**< Flags */
} CUDA_ARRAY3D_DESCRIPTOR;

/**
 * CUDA Resource descriptor
 */
typedef struct CUDA_RESOURCE_DESC_st {
  CUresourcetype resType; /**< Resource type */

  union {
    struct {
      CUarray hArray; /**< CUDA array */
    } array;
    struct {
      CUmipmappedArray hMipmappedArray; /**< CUDA mipmapped array */
    } mipmap;
    struct {
      CUdeviceptr devPtr;       /**< Device pointer */
      CUarray_format format;    /**< Array format */
      unsigned int numChannels; /**< Channels per array element */
      size_t sizeInBytes;       /**< Size in bytes */
    } linear;
    struct {
      CUdeviceptr devPtr;       /**< Device pointer */
      CUarray_format format;    /**< Array format */
      unsigned int numChannels; /**< Channels per array element */
      size_t width;             /**< Width of the array in elements */
      size_t height;            /**< Height of the array in elements */
      size_t pitchInBytes;      /**< Pitch between two rows in bytes */
    } pitch2D;
    struct {
      int reserved[32];
    } reserved;
  } res;

  unsigned int flags; /**< Flags (must be zero) */
} CUDA_RESOURCE_DESC;

typedef enum CUexternalMemoryHandleType_enum {
  /**
   * Handle is an opaque file descriptor
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
  /**
   * Handle is an opaque shared NT handle
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
  /**
   * Handle is an opaque, globally shared handle
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
  /**
   * Handle is a D3D12 heap object
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
  /**
   * Handle is a D3D12 committed resource
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
  /**
   * Handle is a shared NT handle to a D3D11 resource
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
  /**
   * Handle is a globally shared handle to a D3D11 resource
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
  /**
   * Handle is an NvSciBuf object
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
} CUexternalMemoryHandleType;

typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
  /**
   * Type of the handle
   */
  CUexternalMemoryHandleType type;
  union {
    /**
     * File descriptor referencing the memory object. Valid
     * when type is
     * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
     */
    int fd;
    /**
     * Win32 handle referencing the semaphore object. Valid when
     * type is one of the following:
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
     * Exactly one of 'handle' and 'name' must be non-NULL. If
     * type is one of the following:
     * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
     * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
     * then 'name' must be NULL.
     */
    struct {
      /**
       * Valid NT handle. Must be NULL if 'name' is non-NULL
       */
      void *handle;
      /**
       * Name of a valid memory object.
       * Must be NULL if 'handle' is non-NULL.
       */
      const void *name;
    } win32;
    /**
     * A handle representing an NvSciBuf Object. Valid when type
     * is ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
     */
    const void *nvSciBufObject;
  } handle;
  /**
   * Size of the memory allocation
   */
  unsigned long long size;
  /**
   * Flags must either be zero or ::CUDA_EXTERNAL_MEMORY_DEDICATED
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC;

typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
  /**
   * Offset into the memory object where the buffer's base is
   */
  unsigned long long offset;
  /**
   * Size of the buffer
   */
  unsigned long long size;
  /**
   * Flags reserved for future use. Must be zero.
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_BUFFER_DESC;

/**
 * External memory mipmap descriptor
 */
typedef struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st {
  /**
   * Offset into the memory object where the base level of the
   * mipmap chain is.
   */
  unsigned long long offset;
  /**
   * Format, dimension and type of base level of the mipmap chain
   */
  CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
  /**
   * Total number of levels in the mipmap chain
   */
  unsigned int numLevels;
  unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC;

/**
 * External semaphore handle types
 */
typedef enum CUexternalSemaphoreHandleType_enum {
  /**
   * Handle is an opaque file descriptor
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
  /**
   * Handle is an opaque shared NT handle
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
  /**
   * Handle is an opaque, globally shared handle
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
  /**
   * Handle is a shared NT handle referencing a D3D12 fence object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
  /**
   * Handle is a shared NT handle referencing a D3D11 fence object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
  /**
   * Opaque handle to NvSciSync Object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
  /**
   * Handle is a shared NT handle referencing a D3D11 keyed mutex object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
  /**
   * Handle is a globally shared handle referencing a D3D11 keyed mutex object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8
} CUexternalSemaphoreHandleType;

/**
 * External semaphore handle descriptor
 */
typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
  /**
   * Type of the handle
   */
  CUexternalSemaphoreHandleType type;
  union {
    /**
     * File descriptor referencing the semaphore object. Valid
     * when type is
     * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
     */
    int fd;
    /**
     * Win32 handle referencing the semaphore object. Valid when
     * type is one of the following:
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX
     * Exactly one of 'handle' and 'name' must be non-NULL. If
     * type is one of the following:
     * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
     * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
     * then 'name' must be NULL.
     */
    struct {
      /**
       * Valid NT handle. Must be NULL if 'name' is non-NULL
       */
      void *handle;
      /**
       * Name of a valid synchronization primitive.
       * Must be NULL if 'handle' is non-NULL.
       */
      const void *name;
    } win32;
    /**
     * Valid NvSciSyncObj. Must be non NULL
     */
    const void *nvSciSyncObj;
  } handle;
  /**
   * Flags reserved for the future. Must be zero.
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC;

/**
 * External semaphore signal parameters
 */
typedef struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st {
  struct {
    /**
     * Parameters for fence objects
     */
    struct {
      /**
       * Value of fence to be signaled
       */
      unsigned long long value;
    } fence;
    union {
      /**
       * Pointer to NvSciSyncFence. Valid if ::CUexternalSemaphoreHandleType
       * is of type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC.
       */
      void *fence;
      unsigned long long reserved;
    } nvSciSync;
    /**
     * Parameters for keyed mutex objects
     */
    struct {
      /**
       * Value of key to release the mutex with
       */
      unsigned long long key;
    } keyedMutex;
    unsigned int reserved[12];
  } params;
  /**
   * Only when ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to
   * signal a ::CUexternalSemaphore of type
   * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
   * ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which indicates
   * that while signaling the ::CUexternalSemaphore, no memory synchronization
   * operations should be performed for any external memory object imported
   * as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
   * For all other types of ::CUexternalSemaphore, flags must be zero.
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;

/**
 * External semaphore wait parameters
 */
typedef struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st {
  struct {
    /**
     * Parameters for fence objects
     */
    struct {
      /**
       * Value of fence to be waited on
       */
      unsigned long long value;
    } fence;
    /**
     * Pointer to NvSciSyncFence. Valid if CUexternalSemaphoreHandleType
     * is of type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC.
     */
    union {
      void *fence;
      unsigned long long reserved;
    } nvSciSync;
    /**
     * Parameters for keyed mutex objects
     */
    struct {
      /**
       * Value of key to acquire the mutex with
       */
      unsigned long long key;
      /**
       * Timeout in milliseconds to wait to acquire the mutex
       */
      unsigned int timeoutMs;
    } keyedMutex;
    unsigned int reserved[10];
  } params;
  /**
   * Only when ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on
   * a ::CUexternalSemaphore of type
   * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
   * ::CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates that
   * while waiting for the ::CUexternalSemaphore, no memory synchronization
   * operations should be performed for any external memory object imported as
   * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. For all other types of
   * ::CUexternalSemaphore, flags must be zero.
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS;

/**
 * Indicates that the external memory object is a dedicated resource
 */
#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1

/**
 * This flag must be set in order to bind a surface reference
 * to the CUDA array
 */
#define CUDA_ARRAY3D_SURFACE_LDST 0x02

/**
 * This flag indicates that the CUDA array may be bound as a color target
 * in an external graphics API
 */
#define CUDA_ARRAY3D_COLOR_ATTACHMENT 0x20

// copy from cusparse.h
struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;

struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

struct cusparseDnVecDescr;
struct cusparseSpMatDescr;
typedef struct cusparseDnVecDescr *cusparseDnVecDescr_t;
typedef struct cusparseSpMatDescr *cusparseSpMatDescr_t;
typedef enum {
  CUSPARSE_INDEX_16U = 1,  ///< 16-bit unsigned integer for matrix/vector
                           ///< indices
  CUSPARSE_INDEX_32I = 2,  ///< 32-bit signed integer for matrix/vector indices
  CUSPARSE_INDEX_64I = 3   ///< 64-bit signed integer for matrix/vector indices
} cusparseIndexType_t;

typedef enum {
  CUSPARSE_INDEX_BASE_ZERO = 0,
  CUSPARSE_INDEX_BASE_ONE = 1
} cusparseIndexBase_t;

typedef enum cudaDataType_t {
  CUDA_R_16F = 2,   /* real as a half */
  CUDA_C_16F = 6,   /* complex as a pair of half numbers */
  CUDA_R_16BF = 14, /* real as a nv_bfloat16 */
  CUDA_C_16BF = 15, /* complex as a pair of nv_bfloat16 numbers */
  CUDA_R_32F = 0,   /* real as a float */
  CUDA_C_32F = 4,   /* complex as a pair of float numbers */
  CUDA_R_64F = 1,   /* real as a double */
  CUDA_C_64F = 5,   /* complex as a pair of double numbers */
  CUDA_R_4I = 16,   /* real as a signed 4-bit int */
  CUDA_C_4I = 17,   /* complex as a pair of signed 4-bit int numbers */
  CUDA_R_4U = 18,   /* real as a unsigned 4-bit int */
  CUDA_C_4U = 19,   /* complex as a pair of unsigned 4-bit int numbers */
  CUDA_R_8I = 3,    /* real as a signed 8-bit int */
  CUDA_C_8I = 7,    /* complex as a pair of signed 8-bit int numbers */
  CUDA_R_8U = 8,    /* real as a unsigned 8-bit int */
  CUDA_C_8U = 9,    /* complex as a pair of unsigned 8-bit int numbers */
  CUDA_R_16I = 20,  /* real as a signed 16-bit int */
  CUDA_C_16I = 21,  /* complex as a pair of signed 16-bit int numbers */
  CUDA_R_16U = 22,  /* real as a unsigned 16-bit int */
  CUDA_C_16U = 23,  /* complex as a pair of unsigned 16-bit int numbers */
  CUDA_R_32I = 10,  /* real as a signed 32-bit int */
  CUDA_C_32I = 11,  /* complex as a pair of signed 32-bit int numbers */
  CUDA_R_32U = 12,  /* real as a unsigned 32-bit int */
  CUDA_C_32U = 13,  /* complex as a pair of unsigned 32-bit int numbers */
  CUDA_R_64I = 24,  /* real as a signed 64-bit int */
  CUDA_C_64I = 25,  /* complex as a pair of signed 64-bit int numbers */
  CUDA_R_64U = 26,  /* real as a unsigned 64-bit int */
  CUDA_C_64U = 27   /* complex as a pair of unsigned 64-bit int numbers */
} cudaDataType;

typedef enum {
  CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
  CUSPARSE_OPERATION_TRANSPOSE = 1,
  CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

typedef enum {
  CUSPARSE_SPMV_ALG_DEFAULT = 0,
  CUSPARSE_SPMV_COO_ALG1 = 1,
  CUSPARSE_SPMV_CSR_ALG1 = 2,
  CUSPARSE_SPMV_CSR_ALG2 = 3,
  CUSPARSE_SPMV_COO_ALG2 = 4
} cusparseSpMVAlg_t;

typedef enum {
  CUSPARSE_MATRIX_TYPE_GENERAL = 0,
  CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
  CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
  CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} cusparseMatrixType_t;

typedef enum {
  CUSPARSE_FILL_MODE_LOWER = 0,
  CUSPARSE_FILL_MODE_UPPER = 1
} cusparseFillMode_t;

typedef enum {
  CUSPARSE_DIAG_TYPE_NON_UNIT = 0,
  CUSPARSE_DIAG_TYPE_UNIT = 1
} cusparseDiagType_t;

// copy from cusolver.h
typedef enum libraryPropertyType_t {
  MAJOR_VERSION,
  MINOR_VERSION,
  PATCH_LEVEL
} libraryPropertyType;

struct cusolverSpContext;
typedef struct cusolverSpContext *cusolverSpHandle_t;
#endif
