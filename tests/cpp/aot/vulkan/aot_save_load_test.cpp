#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/inc/constants.h"
#include "taichi/program/program.h"
#include "tests/cpp/ir/ndarray_kernel.h"
#include "tests/cpp/program/test_program.h"
#include "taichi/aot/graph_data.h"
#include "taichi/program/graph_builder.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/device.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_utils.h"
#endif

using namespace taichi;
using namespace lang;

[[maybe_unused]] static void aot_save() {
  auto program = Program(Arch::vulkan);

  program.config.advanced_optimization = false;

  int n = 10;

  auto *root = new SNode(0, SNodeType::root);
  auto *pointer = &root->dense(Axis(0), n, false);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/true);

  auto aot_builder = program.make_aot_module_builder(Arch::vulkan);

  std::unique_ptr<Kernel> kernel_init, kernel_ret, kernel_simple_ret;

  {
    /*
    @ti.kernel
    def ret() -> ti.f32:
      sum = 0.2
      return sum
    */
    IRBuilder builder;
    auto *sum = builder.create_local_var(PrimitiveType::f32);
    builder.create_local_store(sum, builder.get_float32(0.2));
    builder.create_return(builder.create_local_load(sum));

    kernel_simple_ret =
        std::make_unique<Kernel>(program, builder.extract_ir(), "simple_ret");
    kernel_simple_ret->insert_ret(PrimitiveType::f32);
  }

  {
    /*
    @ti.kernel
    def init():
      for index in range(n):
        place[index] = index
    */
    IRBuilder builder;
    auto *zero = builder.get_int32(0);
    auto *n_stmt = builder.get_int32(n);
    auto *loop = builder.create_range_for(zero, n_stmt, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ptr = builder.create_global_ptr(place, {index});
      builder.create_global_store(ptr, index);
    }

    kernel_init =
        std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }

  {
    /*
    @ti.kernel
    def ret():
      sum = 0
      for index in place:
        sum = sum + place[index];
      return sum
    */
    IRBuilder builder;
    auto *sum = builder.create_local_var(PrimitiveType::i32);
    auto *loop = builder.create_struct_for(pointer, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *sum_old = builder.create_local_load(sum);
      auto *place_index =
          builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_local_store(sum, builder.create_add(sum_old, place_index));
    }
    builder.create_return(builder.create_local_load(sum));

    kernel_ret = std::make_unique<Kernel>(program, builder.extract_ir(), "ret");
    kernel_ret->insert_ret(PrimitiveType::i32);
  }

  aot_builder->add("simple_ret", kernel_simple_ret.get());
  aot_builder->add_field("place", place, true, place->dt, {n}, 1, 1);
  aot_builder->add("init", kernel_init.get());
  aot_builder->add("ret", kernel_ret.get());
  aot_builder->dump(".", "");
}

[[maybe_unused]] static void save_ndarray_kernels(Arch arch) {
  TestProgram test_prog;
  test_prog.setup(arch);
  auto aot_builder = test_prog.prog()->make_aot_module_builder(arch);
  auto ker1 = setup_kernel1(test_prog.prog());
  auto ker2 = setup_kernel2(test_prog.prog());
  aot_builder->add("ker1", ker1.get());
  aot_builder->add("ker2", ker2.get());
  aot_builder->dump(".", "");
}

#ifdef TI_WITH_VULKAN
[[maybe_unused]] static void write_devalloc(
    taichi::lang::gfx::GfxRuntime *vulkan_runtime,
    taichi::lang::DeviceAllocation &alloc,
    const void *data,
    size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(alloc));
  std::memcpy(device_arr_ptr, data, size);
  vulkan_runtime->get_ti_device()->unmap(alloc);
}

[[maybe_unused]] static void load_devalloc(
    taichi::lang::gfx::GfxRuntime *vulkan_runtime,
    taichi::lang::DeviceAllocation &alloc,
    void *data,
    size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(alloc));
  std::memcpy(data, device_arr_ptr, size);
  vulkan_runtime->get_ti_device()->unmap(alloc);
}

TEST(AotSaveLoad, Vulkan) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  aot_save();

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  // Create Vulkan runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = embedded_device->device();
  auto vulkan_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ".";
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 64);
  vulkan_runtime->add_root_buffer(root_size);

  auto simple_ret_kernel = vk_module->get_kernel("simple_ret");
  EXPECT_TRUE(simple_ret_kernel);

  simple_ret_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();
  EXPECT_FLOAT_EQ(host_ctx.get_ret<float>(0), 0.2);

  auto init_kernel = vk_module->get_kernel("init");
  EXPECT_TRUE(init_kernel);

  auto ret_kernel = vk_module->get_kernel("ret");
  EXPECT_TRUE(ret_kernel);

  auto ret2_kernel = vk_module->get_kernel("ret2");
  EXPECT_FALSE(ret2_kernel);

  // Run kernels
  init_kernel->launch(&host_ctx);
  ret_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();

  // Retrieve data
  auto x_field = vk_module->get_snode_tree("place");
  EXPECT_NE(x_field, nullptr);
}

TEST(AotSaveLoad, VulkanNdarray) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  save_ndarray_kernels(Arch::vulkan);

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  // Create Vulkan runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = embedded_device->device();
  auto vulkan_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ".";
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 0);
  vulkan_runtime->add_root_buffer(root_size);

  auto ker1 = vk_module->get_kernel("ker1");
  EXPECT_TRUE(ker1);

  const int size = 10;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.host_read = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ =
      embedded_device->device()->allocate_memory(alloc_params);
  Ndarray arr = Ndarray(devalloc_arr_, PrimitiveType::i32, {size});
  host_ctx.set_arg_ndarray(0, arr.get_device_allocation_ptr_as_int(),
                           arr.shape);
  int src[size] = {0};
  src[0] = 2;
  src[2] = 40;
  write_devalloc(vulkan_runtime.get(), devalloc_arr_, src, sizeof(src));
  ker1->launch(&host_ctx);
  vulkan_runtime->synchronize();

  int dst[size] = {33};
  load_devalloc(vulkan_runtime.get(), devalloc_arr_, dst, sizeof(dst));
  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 1);
  EXPECT_EQ(dst[2], 42);

  auto ker2 = vk_module->get_kernel("ker2");
  EXPECT_TRUE(ker2);

  host_ctx.set_arg(1, 3);
  ker2->launch(&host_ctx);
  vulkan_runtime->synchronize();
  load_devalloc(vulkan_runtime.get(), devalloc_arr_, dst, sizeof(dst));
  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 3);
  EXPECT_EQ(dst[2], 42);

  // Deallocate
  embedded_device->device()->dealloc_memory(devalloc_arr_);
}

[[maybe_unused]] static void save_graph() {
  TestProgram test_prog;
  test_prog.setup(Arch::vulkan);
  auto aot_builder = test_prog.prog()->make_aot_module_builder(Arch::vulkan);
  auto ker1 = setup_kernel1(test_prog.prog());
  auto ker2 = setup_kernel2(test_prog.prog());

  auto g_builder = std::make_unique<GraphBuilder>();
  auto seq = g_builder->seq();
  auto arr_arg = aot::Arg{aot::ArgKind::kNdarray, "arr", PrimitiveType::i32, 1};
  seq->dispatch(ker1.get(), {arr_arg});
  seq->dispatch(ker2.get(), {arr_arg, aot::Arg{aot::ArgKind::kScalar, "x",
                                               PrimitiveType::i32}});
  auto graph = g_builder->compile();

  aot_builder->add_graph("test", *graph);
  aot_builder->dump(".", "");
}

TEST(AotLoadGraph, Vulkan) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  save_graph();

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  // Create Vulkan runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device_;
  auto vulkan_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ".";
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 0);
  vulkan_runtime->add_root_buffer(root_size);

  auto graph = vk_module->get_graph("test");

  const int size = 10;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ = device_->allocate_memory(alloc_params);

  int src[size] = {0};
  src[0] = 2;
  src[2] = 40;
  write_devalloc(vulkan_runtime.get(), devalloc_arr_, src, sizeof(src));

  std::unordered_map<std::string, aot::IValue> args;
  auto arr = Ndarray(devalloc_arr_, PrimitiveType::i32, {size});
  args.insert({"arr", aot::IValue::create(arr)});
  args.insert({"x", aot::IValue::create<int>(2)});
  graph->run(args);
  vulkan_runtime->synchronize();

  int dst[size] = {1};
  load_devalloc(vulkan_runtime.get(), devalloc_arr_, dst, sizeof(dst));

  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 2);
  EXPECT_EQ(dst[2], 42);
  device_->dealloc_memory(devalloc_arr_);
}

TEST(AotLoadGraph, Mpm88) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }
  constexpr int NR_PARTICLES = 8192;
  constexpr int N_GRID = 128;

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  // Create Vulkan runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device_;
  auto vulkan_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 0);
  vulkan_runtime->add_root_buffer(root_size);

  auto g_init = vk_module->get_graph("init");
  auto g_update = vk_module->get_graph("update");

  // Prepare Ndarray for model
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = false;
  alloc_params.host_read = false;
  alloc_params.size = NR_PARTICLES * 2 * sizeof(float);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;

  taichi::lang::DeviceAllocation devalloc_x =
      device_->allocate_memory(alloc_params);
  auto x = taichi::lang::Ndarray(devalloc_x, taichi::lang::PrimitiveType::f32,
                                 {NR_PARTICLES}, {2});

  taichi::lang::DeviceAllocation devalloc_v =
      device_->allocate_memory(alloc_params);
  auto v = taichi::lang::Ndarray(devalloc_v, taichi::lang::PrimitiveType::f32,
                                 {NR_PARTICLES}, {2});

  alloc_params.size = NR_PARTICLES * 3 * sizeof(float);
  taichi::lang::DeviceAllocation devalloc_pos =
      device_->allocate_memory(alloc_params);
  auto pos = taichi::lang::Ndarray(
      devalloc_pos, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});

  alloc_params.size = NR_PARTICLES * sizeof(float) * 2 * 2;
  taichi::lang::DeviceAllocation devalloc_C =
      device_->allocate_memory(alloc_params);
  auto C = taichi::lang::Ndarray(devalloc_C, taichi::lang::PrimitiveType::f32,
                                 {NR_PARTICLES}, {2, 2});

  alloc_params.size = NR_PARTICLES * sizeof(float);
  taichi::lang::DeviceAllocation devalloc_J =
      device_->allocate_memory(alloc_params);
  auto J = taichi::lang::Ndarray(devalloc_J, taichi::lang::PrimitiveType::f32,
                                 {NR_PARTICLES});

  alloc_params.size = N_GRID * N_GRID * 2 * sizeof(float);
  taichi::lang::DeviceAllocation devalloc_grid_v =
      device_->allocate_memory(alloc_params);
  auto grid_v = taichi::lang::Ndarray(
      devalloc_grid_v, taichi::lang::PrimitiveType::f32, {N_GRID, N_GRID}, {2});

  alloc_params.size = N_GRID * N_GRID * sizeof(float);
  taichi::lang::DeviceAllocation devalloc_grid_m =
      device_->allocate_memory(alloc_params);
  auto grid_m = taichi::lang::Ndarray(
      devalloc_grid_m, taichi::lang::PrimitiveType::f32, {N_GRID, N_GRID});

  std::unordered_map<std::string, taichi::lang::aot::IValue> args;
  args.insert({"x", taichi::lang::aot::IValue::create(x)});
  args.insert({"v", taichi::lang::aot::IValue::create(v)});
  args.insert({"J", taichi::lang::aot::IValue::create(J)});

  g_init->run(args);
  vulkan_runtime->synchronize();

  args.insert({"C", taichi::lang::aot::IValue::create(C)});
  args.insert({"grid_v", taichi::lang::aot::IValue::create(grid_v)});
  args.insert({"grid_m", taichi::lang::aot::IValue::create(grid_m)});
  args.insert({"pos", taichi::lang::aot::IValue::create(pos)});

  // Run update graph once. In real application this runs as long as window is
  // alive.
  g_update->run(args);
  vulkan_runtime->synchronize();

  device_->dealloc_memory(devalloc_x);
  device_->dealloc_memory(devalloc_v);
  device_->dealloc_memory(devalloc_J);
  device_->dealloc_memory(devalloc_C);
  device_->dealloc_memory(devalloc_grid_v);
  device_->dealloc_memory(devalloc_grid_m);
  device_->dealloc_memory(devalloc_pos);
}
#endif
