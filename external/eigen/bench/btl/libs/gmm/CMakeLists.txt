
find_package(GMM)
if (GMM_FOUND)
  include_directories(${GMM_INCLUDES})
  btl_add_bench(btl_gmm main.cpp)
endif (GMM_FOUND)
