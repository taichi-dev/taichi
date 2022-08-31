// C++ wrapper of Taichi C-API
#include <list>
#include <vector>
#include <string>
#include "taichi/taichi.h"

namespace ti {

// Token type for half-precision floats.
struct half {
  uint16_t _;
};

namespace detail {

// Template type to data type enum.
template <typename T>
struct templ2dtype {};
template <>
struct templ2dtype<int8_t> {
  static const TiDataType value = TI_DATA_TYPE_I8;
};
template <>
struct templ2dtype<int16_t> {
  static const TiDataType value = TI_DATA_TYPE_I16;
};
template <>
struct templ2dtype<int32_t> {
  static const TiDataType value = TI_DATA_TYPE_I32;
};
template <>
struct templ2dtype<uint8_t> {
  static const TiDataType value = TI_DATA_TYPE_U8;
};
template <>
struct templ2dtype<uint16_t> {
  static const TiDataType value = TI_DATA_TYPE_U16;
};
template <>
struct templ2dtype<uint32_t> {
  static const TiDataType value = TI_DATA_TYPE_U32;
};
template <>
struct templ2dtype<half> {
  static const TiDataType value = TI_DATA_TYPE_F16;
};
template <>
struct templ2dtype<float> {
  static const TiDataType value = TI_DATA_TYPE_F32;
};
template <>
struct templ2dtype<double> {
  static const TiDataType value = TI_DATA_TYPE_F64;
};

template <typename THandle>
THandle move_handle(THandle &handle) {
  THandle out = std::move(handle);
  handle = TI_NULL_HANDLE;
  return out;
}

}  // namespace detail

class Memory {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiMemory memory_{TI_NULL_HANDLE};
  bool should_destroy_{false};

 public:
  constexpr bool is_valid() const {
    return runtime_ != nullptr;
  }
  inline void destroy() {
    if (should_destroy_) {
      ti_free_memory(runtime_, memory_);
      memory_ = TI_NULL_HANDLE;
      should_destroy_ = false;
    }
  }

  Memory() {
  }
  Memory(const Memory &) = delete;
  Memory(Memory &&b)
      : runtime_(detail::move_handle(b.runtime_)),
        memory_(detail::move_handle(b.memory_)),
        should_destroy_(std::exchange(b.should_destroy_, false)) {
  }
  Memory(TiRuntime runtime, TiMemory memory, bool should_destroy)
      : runtime_(runtime), memory_(memory), should_destroy_(should_destroy) {
  }
  ~Memory() {
    destroy();
  }

  Memory &operator=(const Memory &) = delete;
  Memory &operator=(Memory &&b) {
    destroy();
    runtime_ = detail::move_handle(b.runtime_);
    memory_ = detail::move_handle(b.memory_);
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  constexpr TiMemory memory() const {
    return memory_;
  }
  constexpr operator TiMemory() const {
    return memory_;
  }
};

template <typename T>
class NdArray {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiNdArray ndarray_{};
  bool should_destroy_{false};

 public:
  constexpr bool is_valid() const {
    return ndarray_.memory != nullptr;
  }
  inline void destroy() {
    if (should_destroy_) {
      ti_free_memory(runtime_, ndarray_.memory);
      ndarray_.memory = TI_NULL_HANDLE;
      should_destroy_ = false;
    }
  }

  NdArray() {
  }
  NdArray(const NdArray<T> &) = delete;
  NdArray(NdArray<T> &&b)
      : runtime_(detail::move_handle(b.runtime_)),
        ndarray_(std::exchange(b.ndarray_, {})),
        should_destroy_(std::exchange(b.should_destroy_, false)) {
  }
  NdArray(TiRuntime runtime, const TiNdArray &ndarray, bool should_destroy)
      : runtime_(runtime), ndarray_(ndarray), should_destroy_(should_destroy) {
  }
  ~NdArray() {
    destroy();
  }

  NdArray<T> &operator=(const NdArray<T> &) = delete;
  NdArray<T> &operator=(NdArray<T> &&b) {
    destroy();
    runtime_ = detail::move_handle(b.runtime_);
    ndarray_ = std::exchange(b.ndarray_, {});
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  void *map() {
    return ti_map_memory(runtime_, ndarray_.memory);
  }
  void unmap() {
    return ti_unmap_memory(runtime_, ndarray_.memory);
  }

  constexpr TiMemory memory() const {
    return ndarray_.memory;
  }
  constexpr TiNdArray ndarray() const {
    return ndarray_;
  }
  constexpr operator TiNdArray() const {
    return ndarray_;
  }
};

class Texture {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiTexture texture_{TI_NULL_HANDLE};
  bool should_destroy_{false};

 public:
  constexpr bool is_valid() const {
    return texture_ != nullptr;
  }
  inline void destroy() {
    if (should_destroy_) {
      ti_free_texture(runtime_, texture_);
      texture_ = TI_NULL_HANDLE;
      should_destroy_ = false;
    }
  }

  Texture() {
  }
  Texture(const Texture &b) = delete;
  Texture(Texture &&b)
      : runtime_(detail::move_handle(b.runtime_)),
        texture_(detail::move_handle(b.texture_)),
        should_destroy_(std::exchange(b.should_destroy_, false)) {
  }
  Texture(TiRuntime runtime, TiTexture texture, bool should_destroy)
      : runtime_(runtime), texture_(texture), should_destroy_(should_destroy) {
  }
  ~Texture() {
    destroy();
  }

  Texture &operator=(const Texture &) = delete;
  Texture &operator=(Texture &&b) {
    destroy();
    runtime_ = detail::move_handle(b.runtime_);
    texture_ = detail::move_handle(b.texture_);
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  constexpr TiTexture texture() const {
    return texture_;
  }
  constexpr operator TiTexture() const {
    return texture_;
  }
};

class ArgumentEntry {
  friend class ComputeGraph;
  TiArgument *arg_;

 public:
  ArgumentEntry() = delete;
  ArgumentEntry(const ArgumentEntry &) = delete;
  ArgumentEntry(ArgumentEntry &&) = delete;
  ArgumentEntry(TiArgument *arg) : arg_(arg) {
  }

  inline ArgumentEntry &operator=(const TiArgument &b) {
    *arg_ = b;
    return *this;
  }
  inline ArgumentEntry &operator=(int32_t i32) {
    arg_->type = TI_ARGUMENT_TYPE_I32;
    arg_->value.i32 = i32;
    return *this;
  }
  inline ArgumentEntry &operator=(float f32) {
    arg_->type = TI_ARGUMENT_TYPE_F32;
    arg_->value.f32 = f32;
    return *this;
  }
  inline ArgumentEntry &operator=(const TiNdArray &ndarray) {
    arg_->type = TI_ARGUMENT_TYPE_NDARRAY;
    arg_->value.ndarray = ndarray;
    return *this;
  }
  inline ArgumentEntry &operator=(TiTexture texture) {
    arg_->type = TI_ARGUMENT_TYPE_TEXTURE;
    arg_->value.texture = texture;
    return *this;
  }
};

class ComputeGraph {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiComputeGraph compute_graph_{TI_NULL_HANDLE};
  std::list<std::string> arg_names_{};  // For stable addresses.
  std::vector<TiNamedArgument> args_{};

 public:
  constexpr bool is_valid() const {
    return compute_graph_ != nullptr;
  }

  ComputeGraph() {
  }
  ComputeGraph(const ComputeGraph &) = delete;
  ComputeGraph(ComputeGraph &&b)
      : runtime_(detail::move_handle(b.runtime_)),
        compute_graph_(detail::move_handle(b.compute_graph_)),
        arg_names_(std::move(b.arg_names_)),
        args_(std::move(b.args_)) {
  }
  ComputeGraph(TiRuntime runtime, TiComputeGraph compute_graph)
      : runtime_(runtime), compute_graph_(compute_graph) {
  }
  ~ComputeGraph() {
  }

  ComputeGraph &operator=(const ComputeGraph &) = delete;
  ComputeGraph &operator=(ComputeGraph &&b) {
    runtime_ = detail::move_handle(b.runtime_);
    compute_graph_ = detail::move_handle(b.compute_graph_);
    arg_names_ = std::move(b.arg_names_);
    args_ = std::move(b.args_);
    return *this;
  }

  inline ArgumentEntry at(const char *name) {
    size_t i = 0;
    auto it = arg_names_.begin();
    for (; it != arg_names_.end(); ++it) {
      if (*it == name) {
        break;
      }
      ++i;
    }

    TiArgument *out;
    if (it != arg_names_.end()) {
      out = &args_.at(i).argument;
    } else {
      arg_names_.emplace_back(name);
      args_.emplace_back();
      args_.back().name = arg_names_.back().c_str();
      out = &args_.back().argument;
    }

    return ArgumentEntry(out);
  };
  inline ArgumentEntry at(const std::string &name) {
    return at(name.c_str());
  }
  inline ArgumentEntry operator[](const char *name) {
    return at(name);
  }
  inline ArgumentEntry operator[](const std::string &name) {
    return at(name);
  }

  void launch(size_t argument_count, const TiNamedArgument *arguments) {
    ti_launch_compute_graph(runtime_, compute_graph_, argument_count,
                            arguments);
  }
  void launch() {
    launch(args_.size(), args_.data());
  }
  void launch(const std::vector<TiNamedArgument> &arguments) {
    launch(arguments.size(), arguments.data());
  }

  constexpr TiComputeGraph compute_graph() const {
    return compute_graph_;
  }
  constexpr operator TiComputeGraph() const {
    return compute_graph_;
  }
};

class Kernel {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiKernel kernel_{TI_NULL_HANDLE};
  std::vector<TiArgument> args_{};

 public:
  constexpr bool is_valid() const {
    return kernel_ != nullptr;
  }

  Kernel() {
  }
  Kernel(const Kernel &) = delete;
  Kernel(Kernel &&b)
      : runtime_(detail::move_handle(b.runtime_)),
        kernel_(detail::move_handle(b.kernel_)),
        args_(std::move(b.args_)) {
  }
  Kernel(TiRuntime runtime, TiKernel kernel)
      : runtime_(runtime), kernel_(kernel) {
  }

  Kernel &operator=(const Kernel &) = delete;
  Kernel &operator=(Kernel &&b) {
    runtime_ = detail::move_handle(b.runtime_);
    kernel_ = detail::move_handle(b.kernel_);
    args_ = std::move(b.args_);
    return *this;
  }

  ArgumentEntry at(uint32_t i) {
    if (i < args_.size()) {
      return ArgumentEntry(&args_.at(i));
    } else {
      args_.resize(i + 1);
      return ArgumentEntry(&args_.at(i));
    }
  }
  ArgumentEntry operator[](uint32_t i) {
    return at(i);
  }

  void launch(size_t argument_count, const TiArgument *arguments) {
    ti_launch_kernel(runtime_, kernel_, argument_count, arguments);
  }
  void launch() {
    launch(args_.size(), args_.data());
  }
  void launch(const std::vector<TiArgument> &arguments) {
    launch(arguments.size(), arguments.data());
  }

  constexpr TiKernel kernel() const {
    return kernel_;
  }
  constexpr operator TiKernel() const {
    return kernel_;
  }
};

class AotModule {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiAotModule aot_module_{TI_NULL_HANDLE};
  bool should_destroy_{false};

 public:
  constexpr bool is_valid() const {
    return aot_module_ != nullptr;
  }
  inline void destroy() {
    if (should_destroy_) {
      ti_destroy_aot_module(aot_module_);
      aot_module_ = TI_NULL_HANDLE;
      should_destroy_ = false;
    }
  }

  AotModule() {
  }
  AotModule(const AotModule &) = delete;
  AotModule(AotModule &&b)
      : runtime_(detail::move_handle(b.runtime_)),
        aot_module_(detail::move_handle(b.aot_module_)),
        should_destroy_(std::exchange(b.should_destroy_, false)) {
  }
  AotModule(TiRuntime runtime, TiAotModule aot_module, bool should_destroy)
      : runtime_(runtime),
        aot_module_(aot_module),
        should_destroy_(should_destroy) {
  }
  ~AotModule() {
    destroy();
  }

  AotModule &operator=(const AotModule &) = delete;
  AotModule &operator=(AotModule &&b) {
    runtime_ = detail::move_handle(b.runtime_);
    aot_module_ = detail::move_handle(b.aot_module_);
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  Kernel get_kernel(const char *name) {
    TiKernel kernel_ = ti_get_aot_module_kernel(aot_module_, name);
    return Kernel(runtime_, kernel_);
  }
  ComputeGraph get_compute_graph(const char *name) {
    TiComputeGraph compute_graph_ =
        ti_get_aot_module_compute_graph(aot_module_, name);
    return ComputeGraph(runtime_, compute_graph_);
  }

  constexpr TiAotModule aot_module() const {
    return aot_module_;
  }
  constexpr operator TiAotModule() const {
    return aot_module_;
  }
};

class Event {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiEvent event_{TI_NULL_HANDLE};
  bool should_destroy_{false};

 public:
  constexpr bool is_valid() const {
    return event_ != nullptr;
  }
  inline void destroy() {
    if (should_destroy_) {
      ti_destroy_event(event_);
      event_ = TI_NULL_HANDLE;
      should_destroy_ = false;
    }
  }

  Event() {
  }
  Event(const Event &) = delete;
  Event(Event &&b) : event_(b.event_), should_destroy_(b.should_destroy_) {
  }
  Event(TiRuntime runtime, TiEvent event, bool should_destroy)
      : runtime_(runtime), event_(event), should_destroy_(should_destroy) {
  }
  ~Event() {
    destroy();
  }

  Event &operator=(const Event &) = delete;
  Event &operator=(Event &&b) {
    event_ = detail::move_handle(b.event_);
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  void reset(TiEvent event_) {
    ti_reset_event(runtime_, event_);
  }
  void signal(TiEvent event_) {
    ti_signal_event(runtime_, event_);
  }
  void wait(TiEvent event_) {
    ti_wait_event(runtime_, event_);
  }

  constexpr TiEvent event() const {
    return event_;
  }
  constexpr operator TiEvent() const {
    return event_;
  }
};

class Runtime {
  TiRuntime runtime_{TI_NULL_HANDLE};
  bool should_destroy_{false};

 public:
  constexpr bool is_valid() const {
    return runtime_ != nullptr;
  }
  inline void destroy() {
    if (should_destroy_) {
      ti_destroy_runtime(runtime_);
      runtime_ = TI_NULL_HANDLE;
      should_destroy_ = false;
    }
  }

  Runtime() {
  }
  Runtime(const Runtime &) = delete;
  Runtime(Runtime &&b)
      : runtime_(detail::move_handle(b.runtime_)),
        should_destroy_(std::exchange(b.should_destroy_, false)) {
  }
  Runtime(TiArch arch)
      : runtime_(ti_create_runtime(arch)), should_destroy_(true) {
  }
  Runtime(TiRuntime runtime, bool should_destroy)
      : runtime_(runtime), should_destroy_(should_destroy) {
  }
  ~Runtime() {
    destroy();
  }

  Runtime &operator=(const Runtime &) = delete;
  Runtime &operator=(Runtime &&b) {
    runtime_ = detail::move_handle(b.runtime_);
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  Memory allocate_memory(const TiMemoryAllocateInfo &allocate_info) {
    TiMemory memory = ti_allocate_memory(runtime_, &allocate_info);
    return Memory(runtime_, memory, true);
  }
  Memory allocate_memory(size_t size) {
    TiMemoryAllocateInfo allocate_info{};
    allocate_info.size = size;
    allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    return allocate_memory(allocate_info);
  }
  template <typename T>
  NdArray<T> allocate_ndarray(std::vector<uint32_t> shape,
                              std::vector<uint32_t> elem_shape,
                              bool host_access = false) {
    size_t size = sizeof(T);
    TiNdArray ndarray{};
    for (size_t i = 0; i < shape.size(); ++i) {
      uint32_t x = shape.at(i);
      size *= x;
      ndarray.shape.dims[i] = x;
    }
    ndarray.shape.dim_count = shape.size();
    for (size_t i = 0; i < elem_shape.size(); ++i) {
      uint32_t x = elem_shape.at(i);
      size *= x;
      ndarray.elem_shape.dims[i] = x;
    }
    ndarray.elem_shape.dim_count = elem_shape.size();
    ndarray.elem_type = detail::templ2dtype<T>::value;
    TiMemoryAllocateInfo allocate_info{};
    allocate_info.size = size;
    allocate_info.host_read = host_access;
    allocate_info.host_write = host_access;
    allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    ndarray.memory = ti_allocate_memory(runtime_, &allocate_info);
    return NdArray<T>(runtime_, std::move(ndarray), true);
  }

  Texture allocate_texture(const TiTextureAllocateInfo &allocate_info) {
    TiTexture texture = ti_allocate_texture(runtime_, &allocate_info);
    return Texture(runtime_, texture, true);
  }
  Texture allocate_texture2d(uint32_t width,
                             uint32_t height,
                             TiTextureFormat format) {
    TiTextureAllocateInfo allocate_info{};
    allocate_info.dimension = TI_TEXTURE_DIMENSION_2D;
    allocate_info.extent.array_layer_count = 1;
    allocate_info.extent.width = width;
    allocate_info.extent.height = height;
    allocate_info.extent.depth = 1;
    allocate_info.mip_level_count = 1;
    allocate_info.format = format;
    allocate_info.usage = TI_TEXTURE_USAGE_STORAGE_BIT;
    return allocate_texture(allocate_info);
  }

  AotModule load_aot_module(const char *path) {
    TiAotModule aot_module_ = ti_load_aot_module(runtime_, path);
    return AotModule(runtime_, aot_module_, true);
  }
  AotModule load_aot_module(const std::string &path) {
    return load_aot_module(path.c_str());
  }

  void copy_memory_device_to_device(const TiMemorySlice &dst_memory,
                                    const TiMemorySlice &src_memory) {
    ti_copy_memory_device_to_device(runtime_, &dst_memory, &src_memory);
  }
  void copy_texture_device_to_device(const TiTextureSlice &dst_texture,
                                     const TiTextureSlice &src_texture) {
    ti_copy_texture_device_to_device(runtime_, &dst_texture, &src_texture);
  }
  void transition_texture(TiTexture texture, TiTextureLayout layout) {
    ti_transition_texture(runtime_, texture, layout);
  }

  void submit() {
    ti_submit(runtime_);
  }
  void wait() {
    ti_wait(runtime_);
  }

  constexpr TiRuntime runtime() const {
    return runtime_;
  }
  constexpr operator TiRuntime() const {
    return runtime_;
  }
};

}  // namespace ti
