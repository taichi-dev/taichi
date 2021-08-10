
find_package(Boost)
if (Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  include_directories(${Boost_INCLUDES})
  btl_add_bench(btl_ublas main.cpp)
endif (Boost_FOUND)
