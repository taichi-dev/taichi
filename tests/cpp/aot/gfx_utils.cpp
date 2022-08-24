#include "tests/cpp/aot/gfx_utils.h"

#include "taichi/runtime/gfx/aot_module_loader_impl.h"

namespace taichi {
namespace lang {
namespace aot_test_utils {
static void write_devalloc(taichi::lang::DeviceAllocation &alloc,
                           const void *data,
                           size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(alloc.device->map(alloc));
  std::memcpy(device_arr_ptr, data, size);
  alloc.device->unmap(alloc);
}

static void load_devalloc(taichi::lang::DeviceAllocation &alloc,
                          void *data,
                          size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(alloc.device->map(alloc));
  std::memcpy(data, device_arr_ptr, size);
  alloc.device->unmap(alloc);
}

void view_devalloc_as_ndarray(Device *device_) {
  const int size = 40;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ = device_->allocate_memory(alloc_params);

  std::vector<int> element_shape = {4};
  auto arr1 = Ndarray(devalloc_arr_, PrimitiveType::i32, {10}, element_shape);
  EXPECT_TRUE(arr1.get_element_shape() == element_shape);
  EXPECT_EQ(arr1.total_shape()[0], 10);
  EXPECT_EQ(arr1.total_shape()[1], 4);

  auto arr2 = Ndarray(devalloc_arr_, PrimitiveType::i32, {10}, element_shape,
                      ExternalArrayLayout::kSOA);
  EXPECT_TRUE(arr2.get_element_shape() == element_shape);
  EXPECT_EQ(arr2.total_shape()[0], 4);
  EXPECT_EQ(arr2.total_shape()[1], 10);

  device_->dealloc_memory(devalloc_arr_);
}

void run_dense_field_kernel(Arch arch, taichi::lang::Device *device) {
  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(arch, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device;
  auto gfx_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));
  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = gfx_runtime.get();

  std::unique_ptr<aot::Module> vk_module = aot::Module::load(arch, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 64);
  gfx_runtime->add_root_buffer(root_size);

  auto simple_ret_kernel = vk_module->get_kernel("simple_ret");
  EXPECT_TRUE(simple_ret_kernel);

  simple_ret_kernel->launch(&host_ctx);
  gfx_runtime->synchronize();
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
  gfx_runtime->synchronize();

  // Retrieve data
  auto x_field = vk_module->get_snode_tree("place");
  EXPECT_NE(x_field, nullptr);
}

void run_kernel_test1(Arch arch, taichi::lang::Device *device) {
  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(arch, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device;
  auto gfx_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = gfx_runtime.get();

  std::unique_ptr<aot::Module> vk_module = aot::Module::load(arch, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 0);
  gfx_runtime->add_root_buffer(root_size);

  auto k_run = vk_module->get_kernel("run");

  const int size = 32;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = false;
  alloc_params.host_read = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ = device->allocate_memory(alloc_params);
  Ndarray arr = Ndarray(devalloc_arr_, PrimitiveType::i32, {size});

  host_ctx.set_arg(/*arg_id=*/0, /*base=*/0);
  host_ctx.set_arg_ndarray(/*arg_id=*/1, arr.get_device_allocation_ptr_as_int(),
                           /*shape=*/arr.shape);
  k_run->launch(&host_ctx);
  gfx_runtime->synchronize();

  int dst[size] = {0};
  load_devalloc(devalloc_arr_, dst, sizeof(dst));
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(dst[i], i);
  }

  // Deallocate
  device->dealloc_memory(devalloc_arr_);
}

void run_kernel_test2(Arch arch, taichi::lang::Device *device) {
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(arch, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device;
  auto gfx_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));
  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = gfx_runtime.get();

  std::unique_ptr<aot::Module> vk_module = aot::Module::load(arch, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 0);
  gfx_runtime->add_root_buffer(root_size);

  auto ker1 = vk_module->get_kernel("ker1");
  EXPECT_TRUE(ker1);

  const int size = 10;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.host_read = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ = device->allocate_memory(alloc_params);
  Ndarray arr = Ndarray(devalloc_arr_, PrimitiveType::i32, {size});
  host_ctx.set_arg_ndarray(0, arr.get_device_allocation_ptr_as_int(),
                           arr.shape);
  int src[size] = {0};
  src[0] = 2;
  src[2] = 40;
  write_devalloc(devalloc_arr_, src, sizeof(src));
  ker1->launch(&host_ctx);
  gfx_runtime->synchronize();

  int dst[size] = {33};
  load_devalloc(devalloc_arr_, dst, sizeof(dst));
  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 1);
  EXPECT_EQ(dst[2], 42);

  auto ker2 = vk_module->get_kernel("ker2");
  EXPECT_TRUE(ker2);

  host_ctx.set_arg(1, 3);
  ker2->launch(&host_ctx);
  gfx_runtime->synchronize();
  load_devalloc(devalloc_arr_, dst, sizeof(dst));
  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 3);
  EXPECT_EQ(dst[2], 42);
  // Deallocate
  device->dealloc_memory(devalloc_arr_);
}

void run_cgraph1(Arch arch, taichi::lang::Device *device_) {
  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize  program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(arch, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;
  // Create runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device_;
  auto gfx_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = gfx_runtime.get();

  std::unique_ptr<aot::Module> module = aot::Module::load(arch, mod_params);
  EXPECT_TRUE(module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = module->get_root_size();
  EXPECT_EQ(root_size, 0);
  gfx_runtime->add_root_buffer(root_size);

  // Prepare Ndarray for model
  constexpr int size = 100;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = false;
  alloc_params.host_read = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ = device_->allocate_memory(alloc_params);
  Ndarray arr = Ndarray(devalloc_arr_, PrimitiveType::i32, {size}, {1});

  int base0 = 10;
  int base1 = 20;
  int base2 = 30;

  std::unordered_map<std::string, taichi::lang::aot::IValue> args;
  args.insert({"arr", taichi::lang::aot::IValue::create(arr)});
  args.insert({"base0", taichi::lang::aot::IValue::create(base0)});
  args.insert({"base1", taichi::lang::aot::IValue::create(base1)});
  args.insert({"base2", taichi::lang::aot::IValue::create(base2)});

  // Prepare & Run "init" Graph
  auto graph = module->get_graph("run_graph");
  graph->run(args);
  gfx_runtime->synchronize();

  int dst[size] = {0};
  load_devalloc(devalloc_arr_, dst, sizeof(dst));
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(dst[i], 3 * i + base0 + base1 + base2);
  }

  // Deallocate
  device_->dealloc_memory(devalloc_arr_);
}

void run_cgraph2(Arch arch, taichi::lang::Device *device_) {
  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(arch, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;
  // Create runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device_;
  auto gfx_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = gfx_runtime.get();

  std::unique_ptr<aot::Module> module = aot::Module::load(arch, mod_params);
  EXPECT_TRUE(module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = module->get_root_size();
  EXPECT_EQ(root_size, 0);
  gfx_runtime->add_root_buffer(root_size);

  auto graph = module->get_graph("test");

  const int size = 10;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.host_read = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ = device_->allocate_memory(alloc_params);

  int src[size] = {0};
  src[0] = 2;
  src[2] = 40;
  write_devalloc(devalloc_arr_, src, sizeof(src));

  std::unordered_map<std::string, aot::IValue> args;
  auto arr = Ndarray(devalloc_arr_, PrimitiveType::i32, {size});
  args.insert({"arr", aot::IValue::create(arr)});
  args.insert({"x", aot::IValue::create<int>(2)});
  graph->run(args);
  gfx_runtime->synchronize();

  int dst[size] = {1};
  load_devalloc(devalloc_arr_, dst, sizeof(dst));

  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 2);
  EXPECT_EQ(dst[2], 42);
  device_->dealloc_memory(devalloc_arr_);
}

void run_mpm88_graph(Arch arch, taichi::lang::Device *device_) {
  constexpr int NR_PARTICLES = 8192 * 5;
  constexpr int N_GRID = 128;

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(arch, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;
  // Create runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device_;
  auto gfx_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = gfx_runtime.get();

  std::unique_ptr<aot::Module> module = aot::Module::load(arch, mod_params);
  EXPECT_TRUE(module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = module->get_root_size();
  EXPECT_EQ(root_size, 0);
  gfx_runtime->add_root_buffer(root_size);

  auto g_init = module->get_graph("init");
  auto g_update = module->get_graph("update");

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
  gfx_runtime->synchronize();

  args.insert({"C", taichi::lang::aot::IValue::create(C)});
  args.insert({"grid_v", taichi::lang::aot::IValue::create(grid_v)});
  args.insert({"grid_m", taichi::lang::aot::IValue::create(grid_m)});
  args.insert({"pos", taichi::lang::aot::IValue::create(pos)});

  // Run update graph once. In real application this runs as long as window is
  // alive.
  g_update->run(args);
  gfx_runtime->synchronize();

  device_->dealloc_memory(devalloc_x);
  device_->dealloc_memory(devalloc_v);
  device_->dealloc_memory(devalloc_J);
  device_->dealloc_memory(devalloc_C);
  device_->dealloc_memory(devalloc_grid_v);
  device_->dealloc_memory(devalloc_grid_m);
  device_->dealloc_memory(devalloc_pos);
}
}  // namespace aot_test_utils
}  // namespace lang
}  // namespace taichi
