
find_package(Eigen2)

if(EIGEN2_FOUND)

  include_directories(BEFORE ${EIGEN2_INCLUDE_DIR})
  btl_add_bench(btl_eigen2_linear main_linear.cpp)
  btl_add_bench(btl_eigen2_vecmat main_vecmat.cpp)
  btl_add_bench(btl_eigen2_matmat main_matmat.cpp)
  btl_add_bench(btl_eigen2_adv main_adv.cpp      )

  btl_add_target_property(btl_eigen2_linear COMPILE_FLAGS "-fno-exceptions -DBTL_PREFIX=eigen2")
  btl_add_target_property(btl_eigen2_vecmat COMPILE_FLAGS "-fno-exceptions -DBTL_PREFIX=eigen2")
  btl_add_target_property(btl_eigen2_matmat COMPILE_FLAGS "-fno-exceptions -DBTL_PREFIX=eigen2")
  btl_add_target_property(btl_eigen2_adv    COMPILE_FLAGS "-fno-exceptions -DBTL_PREFIX=eigen2")

  btl_add_bench(btl_tiny_eigen2 btl_tiny_eigen2.cpp OFF)

endif() # EIGEN2_FOUND
