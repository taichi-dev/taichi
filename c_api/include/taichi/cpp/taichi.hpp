// C++ wrapper of Taichi C-API
#pragma once
#include <cstddef>
#include <cstring>
#include <list>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "taichi/taichi.h"

namespace ti {

inline std::vector<TiArch> get_available_archs() {
  uint32_t narch = 0;
  ti_get_available_archs(&narch, nullptr);
  std::vector<TiArch> archs(narch);
  ti_get_available_archs(&narch, archs.data());
  return archs;
}
inline bool is_arch_available(TiArch arch) {
  std::vector<TiArch> archs = get_available_archs();
  for (size_t i = 0; i < archs.size(); ++i) {
    if (archs.at(i) == arch) {
      return true;
    }
  }
  return false;
}

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
  size_t size_{0};
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
        size_(std::exchange(b.size_, 0)),
        should_destroy_(std::exchange(b.should_destroy_, false)) {
  }
  Memory(TiRuntime runtime, TiMemory memory, size_t size, bool should_destroy)
      : runtime_(runtime),
        memory_(memory),
        size_(size),
        should_destroy_(should_destroy) {
  }
  ~Memory() {
    destroy();
  }

  Memory &operator=(const Memory &) = delete;
  Memory &operator=(Memory &&b) {
    destroy();
    runtime_ = detail::move_handle(b.runtime_);
    memory_ = detail::move_handle(b.memory_);
    size_ = std::exchange(b.size_, 0);
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  void *map() const {
    return ti_map_memory(runtime_, memory_);
  }
  void unmap() const {
    ti_unmap_memory(runtime_, memory_);
  }

  inline void read(void *dst, size_t size) const {
    void *src = map();
    if (src != nullptr) {
      std::memcpy(dst, src, size);
    }
    unmap();
  }
  inline void write(const void *src, size_t size) const {
    void *dst = map();
    if (dst != nullptr) {
      std::memcpy(dst, src, size);
    }
    unmap();
  }

  TiMemorySlice slice(size_t offset, size_t size) const {
    if (offset + size > size_) {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "size");
      return {};
    }
    TiMemorySlice slice{};
    slice.memory = memory_;
    slice.offset = offset;
    slice.size = size;
    return slice;
  }
  TiMemorySlice slice() const {
    return slice(0, size_);
  }

  constexpr size_t size() const {
    return size_;
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
  Memory memory_{};
  TiNdArray ndarray_{};
  size_t elem_count_{};
  size_t scalar_count_{};

 public:
  constexpr bool is_valid() const {
    return memory_.is_valid();
  }
  inline void destroy() {
    memory_.destroy();
  }

  NdArray() : elem_count_(1), scalar_count_(1) {
  }
  NdArray(const NdArray<T> &) = delete;
  NdArray(NdArray<T> &&b)
      : memory_(std::move(b.memory_)),
        ndarray_(std::exchange(b.ndarray_, {})),
        elem_count_(std::exchange(b.elem_count_, 1)),
        scalar_count_(std::exchange(b.scalar_count_, 1)) {
  }
  NdArray(Memory &&memory, const TiNdArray &ndarray)
      : memory_(std::move(memory)),
        ndarray_(ndarray),
        elem_count_(1),
        scalar_count_(1) {
    if (ndarray.memory != memory_) {
      ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, "ndarray.memory != memory");
    }
    for (uint32_t i = 0; i < ndarray_.shape.dim_count; ++i) {
      elem_count_ *= ndarray_.shape.dims[i];
    }
    scalar_count_ *= elem_count_;
    for (uint32_t i = 0; i < ndarray_.elem_shape.dim_count; ++i) {
      scalar_count_ *= ndarray_.elem_shape.dims[i];
    }
  }
  ~NdArray() {
    destroy();
  }

  NdArray<T> &operator=(const NdArray<T> &) = delete;
  NdArray<T> &operator=(NdArray<T> &&b) {
    destroy();
    memory_ = std::move(b.memory_);
    ndarray_ = std::exchange(b.ndarray_, {});
    elem_count_ = std::exchange(b.elem_count_, 1);
    scalar_count_ = std::exchange(b.scalar_count_, 1);
    return *this;
  }

  inline void *map() const {
    return memory_.map();
  }
  inline void unmap() const {
    return memory_.unmap();
  }

  inline size_t scalar_count() const {
    return scalar_count_;
  }
  inline size_t elem_count() const {
    return elem_count_;
  }

  inline void read(T *dst, size_t count) const {
    if (count > scalar_count_) {
      ti_set_last_error(
          TI_ERROR_ARGUMENT_OUT_OF_RANGE,
          "ndarray read ouf of range; please ensure you specified the correct "
          "number of elements (rather than size-in-bytes) to be read");
      return;
    }
    memory_.read(dst, count * sizeof(T));
  }
  inline void read(std::vector<T> &dst) const {
    read(dst.data(), dst.size());
  }
  template <typename U>
  inline void read(std::vector<U> &dst) const {
    static_assert(sizeof(U) % sizeof(T) == 0,
                  "sizeof(U) must be a multiple of sizeof(T)");
    read((T *)dst.data(), dst.size() * (sizeof(U) / sizeof(T)));
  }
  inline void write(const T *src, size_t count) const {
    if (count > scalar_count_) {
      ti_set_last_error(
          TI_ERROR_ARGUMENT_OUT_OF_RANGE,
          "ndarray write ouf of range; please ensure you specified the correct "
          "number of elements (rather than size-in-bytes) to be written");
      return;
    }
    memory_.write(src, count * sizeof(T));
  }
  inline void write(const std::vector<T> &src) const {
    write(src.data(), src.size());
  }
  template <typename U>
  inline void write(const std::vector<U> &src) const {
    static_assert(sizeof(U) % sizeof(T) == 0,
                  "sizeof(U) must be a multiple of sizeof(T)");
    write((const T *)src.data(), src.size() * (sizeof(U) / sizeof(T)));
  }

  TiMemorySlice slice(size_t offset, size_t size) const {
    return memory_.slice(offset, size);
  }
  TiMemorySlice slice() const {
    return memory_.slice();
  }

  constexpr TiDataType elem_type() const {
    return ndarray_.elem_type;
  }
  constexpr const TiNdShape &shape() const {
    return ndarray_.shape;
  }
  constexpr const TiNdShape &elem_shape() const {
    return ndarray_.elem_shape;
  }
  constexpr const Memory &memory() const {
    return memory_;
  }
  constexpr const TiNdArray &ndarray() const {
    return ndarray_;
  }
  constexpr operator TiNdArray() const {
    return ndarray_;
  }
};

class Image {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiImage image_{TI_NULL_HANDLE};
  bool should_destroy_{false};

 public:
  constexpr bool is_valid() const {
    return image_ != nullptr;
  }
  inline void destroy() {
    if (should_destroy_) {
      ti_free_image(runtime_, image_);
      image_ = TI_NULL_HANDLE;
      should_destroy_ = false;
    }
  }

  Image() {
  }
  Image(const Image &b) = delete;
  Image(Image &&b)
      : runtime_(detail::move_handle(b.runtime_)),
        image_(detail::move_handle(b.image_)),
        should_destroy_(std::exchange(b.should_destroy_, false)) {
  }
  Image(TiRuntime runtime, TiImage image, bool should_destroy)
      : runtime_(runtime), image_(image), should_destroy_(should_destroy) {
  }
  ~Image() {
    destroy();
  }

  Image &operator=(const Image &) = delete;
  Image &operator=(Image &&b) {
    destroy();
    runtime_ = detail::move_handle(b.runtime_);
    image_ = detail::move_handle(b.image_);
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  TiImageSlice slice(TiImageOffset offset,
                     TiImageExtent extent,
                     uint32_t mip_level) const {
    TiImageSlice slice{};
    slice.image = image_;
    slice.extent = extent;
    slice.offset = offset;
    slice.mip_level = mip_level;
    return slice;
  }

  constexpr TiImage image() const {
    return image_;
  }
  constexpr operator TiImage() const {
    return image_;
  }
};

class Texture {
  Image image_{};
  TiTexture texture_{};

 public:
  constexpr bool is_valid() const {
    return image_.is_valid();
  }
  inline void destroy() {
    image_.destroy();
  }

  Texture() {
  }
  Texture(const Texture &b) = delete;
  Texture(Texture &&b)
      : image_(std::move(b.image_)), texture_(std::move(b.texture_)) {
  }
  Texture(Image &&image, const TiTexture &texture)
      : image_(std::move(image)), texture_(texture) {
    if (texture.image != image_) {
      ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, "texture.image != image");
    }
  }
  ~Texture() {
    destroy();
  }

  Texture &operator=(const Texture &) = delete;
  Texture &operator=(Texture &&b) {
    destroy();
    image_ = std::move(b.image_);
    texture_ = std::move(b.texture_);
    return *this;
  }

  constexpr const Image &image() const {
    return image_;
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
  ArgumentEntry(ArgumentEntry &&b) : arg_(b.arg_) {
  }
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
  inline ArgumentEntry &operator=(const TiTexture &texture) {
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

  void launch(uint32_t argument_count, const TiNamedArgument *arguments) {
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

  template <typename T>
  void push_arg(const std::vector<T> &v) {
    int idx = args_.size();
    // Temporary workaround for setting vec/matrix arguments in a flattened way.
    args_.resize(args_.size() + v.size());
    for (int j = 0; j < v.size(); ++j) {
      at(idx + j) = v[j];
    }
  }

  template <typename T>
  void push_arg(const T &arg) {
    int idx = args_.size();
    args_.resize(idx + 1);
    at(idx) = arg;
  }

  void clear_args() {
    args_.clear();
  }

  void launch(uint32_t argument_count, const TiArgument *arguments) {
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

class CapabilityLevelConfigBuilder;
class CapabilityLevelConfig {
 public:
  std::vector<TiCapabilityLevelInfo> cap_level_infos;

  CapabilityLevelConfig() : cap_level_infos() {
  }
  CapabilityLevelConfig(std::vector<TiCapabilityLevelInfo> &&capabilities)
      : cap_level_infos(std::move(capabilities)) {
  }

  static CapabilityLevelConfigBuilder builder();

  uint32_t get(TiCapability capability) const {
    for (size_t i = 0; i < cap_level_infos.size(); ++i) {
      const TiCapabilityLevelInfo &cap_level_info = cap_level_infos.at(i);
      if (cap_level_info.capability == capability) {
        return cap_level_info.level;
      }
    }
    return 0;
  }

  void set(TiCapability capability, uint32_t level) {
    std::vector<TiCapabilityLevelInfo>::iterator it = cap_level_infos.begin();
    for (; it != cap_level_infos.end(); ++it) {
      if (it->capability == capability) {
        it->level = level;
        return;
      }
    }
    TiCapabilityLevelInfo cap_level_info{};
    cap_level_info.capability = capability;
    cap_level_info.level = level;
    cap_level_infos.emplace_back(std::move(cap_level_info));
  }
};

class CapabilityLevelConfigBuilder {
  typedef CapabilityLevelConfigBuilder Self;
  std::map<TiCapability, uint32_t> cap_level_infos;

 public:
  CapabilityLevelConfigBuilder() : cap_level_infos() {
  }
  CapabilityLevelConfigBuilder(const Self &) = delete;
  Self &operator=(const Self &) = delete;

  Self &spirv_version(uint32_t major, uint32_t minor) {
    if (major == 1) {
      if (minor == 3) {
        cap_level_infos[TI_CAPABILITY_SPIRV_VERSION] = 0x10300;
      } else if (minor == 4) {
        cap_level_infos[TI_CAPABILITY_SPIRV_VERSION] = 0x10400;
      } else if (minor == 5) {
        cap_level_infos[TI_CAPABILITY_SPIRV_VERSION] = 0x10500;
      } else {
        ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "minor");
      }
    } else {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "major");
    }
    return *this;
  }
  Self &spirv_has_int8(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_INT8] = value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_int16(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_INT16] = value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_int64(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_INT64] = value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_float16(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_FLOAT16] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_float64(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_FLOAT64] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_i64(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_ATOMIC_I64] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float16(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float16_add(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16_ADD] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float16_minmax(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16_MINMAX] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float64(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float64_add(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_ADD] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_variable_ptr(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_VARIABLE_PTR] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_physical_storage_buffer(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_PHYSICAL_STORAGE_BUFFER] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_subgroup_basic(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_SUBGROUP_BASIC] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_subgroup_vote(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_SUBGROUP_VOTE] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_subgroup_arithmetic(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_SUBGROUP_ARITHMETIC] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_subgroup_ballot(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_SUBGROUP_BALLOT] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_non_semantic_info(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_NON_SEMANTIC_INFO] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_no_integer_wrap_decoration(bool value = true) {
    cap_level_infos[TI_CAPABILITY_SPIRV_HAS_NO_INTEGER_WRAP_DECORATION] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }

  CapabilityLevelConfig build() {
    std::vector<TiCapabilityLevelInfo> out{};
    for (const auto &pair : cap_level_infos) {
      TiCapabilityLevelInfo cap_level_info{};
      cap_level_info.capability = pair.first;
      cap_level_info.level = pair.second;
      out.emplace_back(std::move(cap_level_info));
    }
    return CapabilityLevelConfig{std::move(out)};
  }
};

inline CapabilityLevelConfigBuilder CapabilityLevelConfig::builder() {
  return {};
}

class Runtime {
  TiArch arch_{TI_ARCH_MAX_ENUM};
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
      : arch_(std::exchange(b.arch_, TI_ARCH_MAX_ENUM)),
        runtime_(detail::move_handle(b.runtime_)),
        should_destroy_(std::exchange(b.should_destroy_, false)) {
  }
  Runtime(TiArch arch)
      : arch_(arch), runtime_(ti_create_runtime(arch, 0)), should_destroy_(true) {
  }
  Runtime(TiArch arch, TiRuntime runtime, bool should_destroy)
      : arch_(arch), runtime_(runtime), should_destroy_(should_destroy) {
  }
  ~Runtime() {
    destroy();
  }

  Runtime &operator=(const Runtime &) = delete;
  Runtime &operator=(Runtime &&b) {
    arch_ = std::exchange(b.arch_, TI_ARCH_MAX_ENUM);
    runtime_ = detail::move_handle(b.runtime_);
    should_destroy_ = std::exchange(b.should_destroy_, false);
    return *this;
  }

  void set_capabilities_ext(
      const std::vector<TiCapabilityLevelInfo> &capabilities) {
    ti_set_runtime_capabilities_ext(runtime_, (uint32_t)capabilities.size(),
                                    capabilities.data());
  }
  void set_capabilities_ext(const CapabilityLevelConfig &capabilities) {
    set_capabilities_ext(capabilities.cap_level_infos);
  }
  CapabilityLevelConfig get_capabilities() const {
    uint32_t n = 0;
    ti_get_runtime_capabilities(runtime_, &n, nullptr);
    std::vector<TiCapabilityLevelInfo> devcaps(n);
    ti_get_runtime_capabilities(runtime_, &n, devcaps.data());
    return CapabilityLevelConfig{std::move(devcaps)};
  }

  Memory allocate_memory(const TiMemoryAllocateInfo &allocate_info) {
    TiMemory memory = ti_allocate_memory(runtime_, &allocate_info);
    return Memory(runtime_, memory, allocate_info.size, true);
  }
  Memory allocate_memory(size_t size) {
    TiMemoryAllocateInfo allocate_info{};
    allocate_info.size = size;
    allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    return allocate_memory(allocate_info);
  }
  template <typename T>
  NdArray<T> allocate_ndarray(const std::vector<uint32_t> &shape = {},
                              const std::vector<uint32_t> &elem_shape = {},
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
    Memory memory = allocate_memory(allocate_info);
    ndarray.memory = memory;
    return NdArray<T>(std::move(memory), ndarray);
  }

  Image allocate_image(const TiImageAllocateInfo &allocate_info) {
    TiImage image = ti_allocate_image(runtime_, &allocate_info);
    return Image(runtime_, image, true);
  }
  Texture allocate_texture2d(uint32_t width,
                             uint32_t height,
                             TiFormat format,
                             TiSampler sampler) {
    TiImageExtent extent{};
    extent.width = width;
    extent.height = height;
    extent.depth = 1;
    extent.array_layer_count = 1;

    TiImageAllocateInfo allocate_info{};
    allocate_info.dimension = TI_IMAGE_DIMENSION_2D;
    allocate_info.extent = extent;
    allocate_info.mip_level_count = 1;
    allocate_info.format = format;
    allocate_info.usage =
        TI_IMAGE_USAGE_STORAGE_BIT | TI_IMAGE_USAGE_SAMPLED_BIT;

    Image image = allocate_image(allocate_info);
    TiTexture texture{};
    texture.image = image;
    texture.dimension = TI_IMAGE_DIMENSION_2D;
    texture.extent = extent;
    texture.format = format;
    texture.sampler = sampler;
    return Texture(std::move(image), texture);
  }

  AotModule load_aot_module(const char *path) {
    TiAotModule aot_module_ = ti_load_aot_module(runtime_, path);
    return AotModule(runtime_, aot_module_, true);
  }
  AotModule load_aot_module(const std::string &path) {
    return load_aot_module(path.c_str());
  }

  AotModule create_aot_module(const void *tcm, size_t size) {
    TiAotModule aot_module = ti_create_aot_module(runtime_, tcm, size);
    return AotModule(runtime_, aot_module, true);
  }
  AotModule create_aot_module(const std::vector<uint8_t> &tcm) {
    return create_aot_module(tcm.data(), tcm.size());
  }

  void copy_memory_device_to_device(const TiMemorySlice &dst_memory,
                                    const TiMemorySlice &src_memory) {
    ti_copy_memory_device_to_device(runtime_, &dst_memory, &src_memory);
  }
  void copy_image_device_to_device(const TiImageSlice &dst_texture,
                                   const TiImageSlice &src_texture) {
    ti_copy_image_device_to_device(runtime_, &dst_texture, &src_texture);
  }
  void transition_image(TiImage image, TiImageLayout layout) {
    ti_transition_image(runtime_, image, layout);
  }

  void flush() {
    ti_flush(runtime_);
  }
  void wait() {
    ti_wait(runtime_);
  }

  constexpr TiArch arch() const {
    return arch_;
  }
  constexpr TiRuntime runtime() const {
    return runtime_;
  }
  constexpr operator TiRuntime() const {
    return runtime_;
  }
};

}  // namespace ti
