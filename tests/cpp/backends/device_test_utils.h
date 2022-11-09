#pragma once
#include "gtest/gtest.h"

#include "taichi/rhi/device.h"
#include "taichi/program/program_impl.h"

namespace device_test_utils {

void test_memory_allocation(taichi::lang::Device *device);

void test_view_devalloc_as_ndarray(taichi::lang::Device *device_);

void test_program(taichi::lang::ProgramImpl *program, taichi::Arch arch);

};  // namespace device_test_utils
