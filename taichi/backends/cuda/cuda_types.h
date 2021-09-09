#pragma once

#if defined(TI_WITH_CUDA_TOOLKIT)

#include <cuda.h>

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
  CU_RESOURCE_TYPE_ARRAY = 0x00,           /**< Array resoure */
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

#endif
