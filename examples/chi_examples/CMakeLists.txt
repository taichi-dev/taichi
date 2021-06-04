cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_ISE_NONE -std=c++17 -fsized-deallocation -Wall  -march=nehalem -DTI_ARCH_x64 -DTI_PASS_EXCEPTION_TO_PYTHON -DTI_INCLUDED -DTI_WITH_CUDA -DTI_WITH_OPENGL -DTI_WITH_CC -O3 -DNDEBUG -static-libgcc -static-libstdc++ -Wl,--wrap=log2f -rdynamic")
set(CHI_EXAMPLES chi_examples)

project(chi_examples)

add_executable(${CHI_EXAMPLES} main.cpp)
target_include_directories(${CHI_EXAMPLES} 
    PUBLIC $ENV{TAICHI_REPO_DIR}
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/include
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/spdlog/include
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/glad/include
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/glfw/include
)
target_link_libraries(${CHI_EXAMPLES} $ENV{TAICHI_REPO_DIR}/build/libtaichi_export_core.so)
