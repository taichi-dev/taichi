

if((NOT TENSOR_INCLUDE_DIR) AND Eigen_SOURCE_DIR)
  # unless TENSOR_INCLUDE_DIR is defined, let's use current Eigen version
  set(TENSOR_INCLUDE_DIR ${Eigen_SOURCE_DIR})
  set(TENSOR_FOUND TRUE)
else()
  find_package(Tensor)
endif()

if (TENSOR_FOUND)

  include_directories(${TENSOR_INCLUDE_DIR})
  btl_add_bench(btl_tensor_linear main_linear.cpp)
  btl_add_bench(btl_tensor_vecmat main_vecmat.cpp)
  btl_add_bench(btl_tensor_matmat main_matmat.cpp)

  btl_add_target_property(btl_tensor_linear COMPILE_FLAGS "-fno-exceptions -DBTL_PREFIX=tensor")
  btl_add_target_property(btl_tensor_vecmat COMPILE_FLAGS "-fno-exceptions -DBTL_PREFIX=tensor")
  btl_add_target_property(btl_tensor_matmat COMPILE_FLAGS "-fno-exceptions -DBTL_PREFIX=tensor")

  option(BTL_BENCH_NOGCCVEC "also bench Eigen explicit vec without GCC's auto vec" OFF)
  if(CMAKE_COMPILER_IS_GNUCXX AND BTL_BENCH_NOGCCVEC)
    btl_add_bench(btl_tensor_nogccvec_linear main_linear.cpp)
    btl_add_bench(btl_tensor_nogccvec_vecmat main_vecmat.cpp)
    btl_add_bench(btl_tensor_nogccvec_matmat main_matmat.cpp)

    btl_add_target_property(btl_tensor_nogccvec_linear COMPILE_FLAGS "-fno-exceptions -fno-tree-vectorize -DBTL_PREFIX=tensor_nogccvec")
    btl_add_target_property(btl_tensor_nogccvec_vecmat COMPILE_FLAGS "-fno-exceptions -fno-tree-vectorize -DBTL_PREFIX=tensor_nogccvec")
    btl_add_target_property(btl_tensor_nogccvec_matmat COMPILE_FLAGS "-fno-exceptions -fno-tree-vectorize -DBTL_PREFIX=tensor_nogccvec")
  endif()


  if(NOT BTL_NOVEC)
    btl_add_bench(btl_tensor_novec_linear main_linear.cpp OFF)
    btl_add_bench(btl_tensor_novec_vecmat main_vecmat.cpp OFF)
    btl_add_bench(btl_tensor_novec_matmat main_matmat.cpp OFF)
    btl_add_target_property(btl_tensor_novec_linear COMPILE_FLAGS "-fno-exceptions -DEIGEN_DONT_VECTORIZE -DBTL_PREFIX=tensor_novec")
    btl_add_target_property(btl_tensor_novec_vecmat COMPILE_FLAGS "-fno-exceptions -DEIGEN_DONT_VECTORIZE -DBTL_PREFIX=tensor_novec")
    btl_add_target_property(btl_tensor_novec_matmat COMPILE_FLAGS "-fno-exceptions -DEIGEN_DONT_VECTORIZE -DBTL_PREFIX=tensor_novec")

  endif(NOT BTL_NOVEC)

endif (TENSOR_FOUND)
