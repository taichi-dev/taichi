#pragma once

#define TC_CHECK_CUDA_ERROR                                        \
  {                                                                \
    auto err = cudaThreadSynchronize();                            \
    if (err) {                                                     \
      printf("File %s line %d, message: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                             \
      exit(-1);                                                    \
    }                                                              \
  }
