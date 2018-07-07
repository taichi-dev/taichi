set(TAICHI_PROJECT_NAME "toys")

file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.h")

include_directories(include)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(TAICHI_PROJECT_LIBRARIES ${TAICHI_PROJECT_LIBRARIES} ${TAICHI_PROJECT_NAME} PARENT_SCOPE)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_library(taichi_pangu SHARED ${PROJECT_SOURCES})
target_link_libraries(taichi_pangu ${CORE_LIBRARY_NAME})
target_link_libraries(taichi_pangu ${MPI_C_LIBRARIES})


