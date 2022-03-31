#
# Copyright (C) The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.10.2)

get_filename_component(root-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.. ABSOLUTE)

# build native_app_glue as a static lib
set(${CMAKE_C_FLAGS}, "${CMAKE_C_FLAGS}")
add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

# Export ANativeActivity_onCreate(),
# Refer to: https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(native-activity SHARED main.cpp)

# Taichi build configuration
include_directories(${root-dir})
include_directories(${root-dir}/external/spdlog/include/)
include_directories(${root-dir}/external/SPIRV-Tools/include/)
include_directories(${root-dir}/external/volk/)
include_directories(${root-dir}/external/VulkanMemoryAllocator/include/)
include_directories(${root-dir}/external/Vulkan-Headers/include/)
include_directories(${root-dir}/external/glm/)
include_directories(${root-dir}/external/eigen/)
include_directories(${root-dir}/taichi/ui/backends/vulkan/)
include_directories(${root-dir}/taichi/backends/vulkan/)
include_directories(${root-dir}/external/imgui/)
include_directories(${root-dir}/external/imgui/backends/)
target_compile_features(native-activity PUBLIC cxx_std_17)
target_compile_options(native-activity PUBLIC -DTI_WITH_VULKAN -DTI_INCLUDED)

target_include_directories(native-activity PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue)

add_library(taichi_export_core SHARED IMPORTED)
set_property(TARGET taichi_export_core PROPERTY IMPORTED_LOCATION "${root-dir}/build/libtaichi_export_core.so")

# add lib dependencies
target_link_libraries(native-activity
    taichi_export_core
    android
    native_app_glue
    log)
