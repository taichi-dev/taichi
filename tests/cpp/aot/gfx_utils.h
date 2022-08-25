#pragma once
#include "gtest/gtest.h"

#include "taichi/rhi/device.h"
#include "taichi/aot/graph_data.h"
#include "taichi/program/graph_builder.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {
namespace aot_test_utils {
[[maybe_unused]] static void write_devalloc(
    taichi::lang::DeviceAllocation &alloc,
    const void *data,
    size_t size);

[[maybe_unused]] static void
load_devalloc(taichi::lang::DeviceAllocation &alloc, void *data, size_t size);

void view_devalloc_as_ndarray(Device *device_);

[[maybe_unused]] void run_cgraph1(Arch arch, taichi::lang::Device *device_);

[[maybe_unused]] void run_cgraph2(Arch arch, taichi::lang::Device *device_);

[[maybe_unused]] void run_kernel_test1(Arch arch, taichi::lang::Device *device);

[[maybe_unused]] void run_kernel_test2(Arch arch, taichi::lang::Device *device);

[[maybe_unused]] void run_dense_field_kernel(Arch arch,
                                             taichi::lang::Device *device);

[[maybe_unused]] void run_mpm88_graph(Arch arch, taichi::lang::Device *device_);
}  // namespace aot_test_utils
}  // namespace lang
}  // namespace taichi
