
find_package(BLAZE)
find_package(Boost COMPONENTS system)
if (BLAZE_FOUND AND Boost_FOUND)
  include_directories(${BLAZE_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
  btl_add_bench(btl_blaze main.cpp)
  # Note: The newest blaze version requires C++14.
  # Ideally, we should set this depending on the version of Blaze we found
  set_property(TARGET btl_blaze PROPERTY CXX_STANDARD 14)
  if(BUILD_btl_blaze)
    target_link_libraries(btl_blaze ${Boost_LIBRARIES})
  endif()
endif ()
