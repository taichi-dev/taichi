#pragma once
#include "gtest/gtest.h"

#include "taichi/rhi/device.h"

namespace device_test_utils {

void test_memory_allocation(taichi::lang::Device* device);

void test_view_devalloc_as_ndarray(taichi::lang::Device *device_);
// [[maybe_unused]] void test_kernel_launch(taichi::lang::Device* device);

};