
find_package(Blitz)

if (BLITZ_FOUND)
  include_directories(${BLITZ_INCLUDES})

  btl_add_bench(btl_blitz btl_blitz.cpp)
  if (BUILD_btl_blitz)
    target_link_libraries(btl_blitz ${BLITZ_LIBRARIES})
  endif (BUILD_btl_blitz)

  btl_add_bench(btl_tiny_blitz btl_tiny_blitz.cpp OFF)
  if (BUILD_btl_tiny_blitz)
    target_link_libraries(btl_tiny_blitz ${BLITZ_LIBRARIES})
  endif (BUILD_btl_tiny_blitz)

endif (BLITZ_FOUND)
