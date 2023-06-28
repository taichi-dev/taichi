// C++ wrapper of Taichi C-API
#pragma once
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <list>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <taichi/taichi.h>

namespace ti {

struct Version {
  uint32_t version;

  explicit Version(uint32_t version) : version(version) {
  }
  Version(uint32_t major, uint32_t minor, uint32_t patch)
      : version((major * 1000 + minor) * 1000 + patch) {
  }
  Version(const Version &) = default;
  Version(Version &&) = default;
  Version &operator=(const Version &) = default;
  Version &operator=(Version &&) = default;

  inline uint32_t major() const {
    return version / 1000000;
  }
  inline uint32_t minor() const {
    return (version / 1000) % 1000;
  }
  inline uint32_t patch() const {
    return version % 1000;
  }
};
inline Version get_version() {
  return Version(ti_get_version());
}

inline std::vector<TiArch> get_available_archs() {
  uint32_t narch = 0;
  ti_get_available_archs(&narch, nullptr);
  std::vector<TiArch> archs(narch);
  ti_get_available_archs(&narch, archs.data());
  return archs;
}
inline std::vector<TiArch> get_available_archs(
    const std::vector<TiArch> &expect_archs) {
  std::vector<TiArch> actual_archs = get_available_archs();
  std::vector<TiArch> out_archs;
  for (TiArch arch : actual_archs) {
    auto it = std::find(expect_archs.begin(), expect_archs.end(), arch);
    if (it != expect_archs.end()) {
      out_archs.emplace_back(arch);
    }
  }
  return out_archs;
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

struct Error {
  TiError error;
  std::string message;

  inline operator TiError() const {  // NOLINT
    return error;
  }
};

inline Error get_last_error() {
  uint64_t message_size = 0;
  ti_get_last_error(&message_size, nullptr);
  std::string message(message_size, '\0');
  TiError error = ti_get_last_error(&message_size, (char *)message.data());
  message.resize(message.size() - 1);
  return Error{error, message};
}
inline void check_last_error() {
  Error error = get_last_error();
  if (error != TI_ERROR_SUCCESS) {
#ifdef TI_WITH_EXCEPTIONS
    throw std::runtime_error(error.message);
#else
    assert(false);
#endif  // TI_WITH_EXCEPTIONS
  }
}
inline void set_last_error(TiError error) {
  ti_set_last_error(error, nullptr);
}
inline void set_last_error(TiError error, const std::string &message) {
  ti_set_last_error(error, message.c_str());
}
inline void set_last_error(const Error &error) {
  set_last_error(error.error, error.message);
}

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
struct templ2dtype<float> {
  static const TiDataType value = TI_DATA_TYPE_F32;
};
template <>
struct templ2dtype<double> {
  static const TiDataType value = TI_DATA_TYPE_F64;
};

template <typename T, typename U>
T exchange(T &storage, U &&value) {
  T out = std::move(storage);
  storage = (T)std::move(value);
  return out;
}

template <typename THandle>
THandle move_handle(THandle &handle) {
  THandle out = std::move(handle);
  handle = TI_NULL_HANDLE;
  return out;
}

}  // namespace detail

class MemorySlice {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiMemorySlice slice_{};

 public:
  MemorySlice() = default;
  MemorySlice(TiRuntime runtime, const TiMemorySlice &slice)
      : runtime_(runtime), slice_(slice) {
  }
  MemorySlice(const MemorySlice &) = default;
  MemorySlice(MemorySlice &&) = default;
  MemorySlice &operator=(const MemorySlice &) = default;
  MemorySlice &operator=(MemorySlice &&) = default;

  inline void copy_to(const MemorySlice &dst) const {
    if (runtime_ != dst.runtime_) {
      ti_set_last_error(
          TI_ERROR_INVALID_ARGUMENT,
          "cannot copy device memory between different runtime instances");
      return;
    }
    if (slice_.size != dst.slice_.size) {
      ti_set_last_error(
          TI_ERROR_INVALID_ARGUMENT,
          "copy source and destination slice must have the same size");
      return;
    }
    ti_copy_memory_device_to_device(runtime_, &dst.slice_, &slice_);
  }

  inline TiMemory memory() const {
    return slice_.memory;
  }
  inline uint64_t offset() const {
    return slice_.offset;
  }
  inline uint64_t size() const {
    return slice_.size;
  }
  inline const TiMemorySlice &slice() const {
    return slice_;
  }
  inline operator const TiMemorySlice &() const {  // NOLINT
    return slice_;
  }
};

class Memory {
 protected:
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
        size_(detail::exchange(b.size_, 0)),
        should_destroy_(detail::exchange(b.should_destroy_, false)) {
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
    size_ = detail::exchange(b.size_, 0);
    should_destroy_ = detail::exchange(b.should_destroy_, false);
    return *this;
  }

  inline Memory borrow() const {
    return Memory(runtime_, memory_, size_, false);
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

  inline void copy_to(const ti::Memory &dst) const {
    slice().copy_to(dst.slice());
  }

  inline MemorySlice slice(size_t offset, size_t size) const {
    if (offset + size > size_) {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "size");
      return {};
    }
    TiMemorySlice slice{};
    slice.memory = memory_;
    slice.offset = offset;
    slice.size = size;
    return MemorySlice(runtime_, slice);
  }
  inline MemorySlice slice() const {
    return slice(0, size_);
  }

  constexpr size_t size() const {
    return size_;
  }
  constexpr TiMemory memory() const {
    return memory_;
  }
  constexpr operator TiMemory() const {  // NOLINT
    return memory_;
  }
};

template <typename T>
class NdArray {
 protected:
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
        ndarray_(detail::exchange(b.ndarray_, TiNdArray{})),
        elem_count_(detail::exchange(b.elem_count_, 1)),
        scalar_count_(detail::exchange(b.scalar_count_, 1)) {
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
    ndarray_ = detail::exchange(b.ndarray_, TiNdArray{});
    elem_count_ = detail::exchange(b.elem_count_, 1);
    scalar_count_ = detail::exchange(b.scalar_count_, 1);
    return *this;
  }

  inline NdArray<T> borrow() const {
    return NdArray<T>(memory_.borrow(), ndarray_);
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

  template <typename U>
  inline void copy_to(const ti::NdArray<U> &dst) const {
    memory().copy_to(dst.memory());
  }

  inline MemorySlice slice() const {
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
  constexpr operator TiNdArray() const {  // NOLINT
    return ndarray_;
  }
};

class ImageSlice {
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiImageSlice slice_{};

 public:
  ImageSlice() = default;
  ImageSlice(TiRuntime runtime, const TiImageSlice &slice)
      : runtime_(runtime), slice_(slice) {
  }
  ImageSlice(const ImageSlice &) = default;
  ImageSlice(ImageSlice &&) = default;
  ImageSlice &operator=(const ImageSlice &) = default;
  ImageSlice &operator=(ImageSlice &&) = default;

  inline void copy_to(const ImageSlice &dst) const {
    if (runtime_ != dst.runtime_) {
      ti_set_last_error(
          TI_ERROR_INVALID_ARGUMENT,
          "cannot copy device memory between different runtime instances");
      return;
    }
    if (slice_.extent.width != dst.slice_.extent.width ||
        slice_.extent.height != dst.slice_.extent.height ||
        slice_.extent.depth != dst.slice_.extent.depth ||
        slice_.extent.array_layer_count !=
            dst.slice_.extent.array_layer_count) {
      ti_set_last_error(
          TI_ERROR_INVALID_ARGUMENT,
          "copy source and destination slice must have the same size");
      return;
    }
    ti_copy_image_device_to_device(runtime_, &dst.slice_, &slice_);
  }

  inline TiImage image() const {
    return slice_.image;
  }
  inline const TiImageOffset &offset() const {
    return slice_.offset;
  }
  inline const TiImageExtent &extent() const {
    return slice_.extent;
  }
  inline const uint32_t &mip_level() const {
    return slice_.mip_level;
  }
  inline const TiImageSlice &slice() const {
    return slice_;
  }
  inline operator TiImageSlice() const {  // NOLINT
    return slice_;
  }
};

class Image {
 protected:
  TiRuntime runtime_{TI_NULL_HANDLE};
  TiImage image_{TI_NULL_HANDLE};
  TiImageDimension dimension_{TI_IMAGE_DIMENSION_MAX_ENUM};
  TiImageExtent extent_{0, 0, 0};
  uint32_t mip_level_count_;
  TiFormat format_{TI_FORMAT_UNKNOWN};
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
        dimension_(detail::exchange(b.dimension_, TI_IMAGE_DIMENSION_MAX_ENUM)),
        extent_(detail::exchange(b.extent_, TiImageExtent{0, 0, 0})),
        mip_level_count_(detail::exchange(b.mip_level_count_, 0)),
        format_(detail::exchange(b.format_, TI_FORMAT_UNKNOWN)),
        should_destroy_(detail::exchange(b.should_destroy_, false)) {
  }
  Image(TiRuntime runtime,
        TiImage image,
        TiImageDimension dimension,
        const TiImageExtent &extent,
        uint32_t mip_level_count,
        TiFormat format,
        bool should_destroy)
      : runtime_(runtime),
        image_(image),
        dimension_(dimension),
        extent_(extent),
        mip_level_count_(mip_level_count),
        format_(format),
        should_destroy_(should_destroy) {
  }
  ~Image() {
    destroy();
  }

  Image &operator=(const Image &) = delete;
  Image &operator=(Image &&b) {
    destroy();
    runtime_ = detail::move_handle(b.runtime_);
    image_ = detail::move_handle(b.image_);
    dimension_ = detail::exchange(b.dimension_, TI_IMAGE_DIMENSION_MAX_ENUM);
    extent_ = detail::exchange(b.extent_, TiImageExtent{0, 0, 0});
    mip_level_count_ = detail::exchange(b.mip_level_count_, 0);
    format_ = detail::exchange(b.format_, TI_FORMAT_UNKNOWN);
    should_destroy_ = detail::exchange(b.should_destroy_, false);
    return *this;
  }

  inline Image borrow() const {
    return Image(runtime_, image_, dimension_, extent_, mip_level_count_,
                 format_, false);
  }

  inline void copy_to(const Image &dst) const {
    slice().copy_to(dst.slice());
  }

  inline void transition_to(TiImageLayout layout) const {
    ti_transition_image(runtime_, image_, layout);
  }

  inline ImageSlice slice(const TiImageOffset &offset,
                          const TiImageExtent &extent,
                          uint32_t mip_level) const {
    if (offset.x + extent.width > extent_.width ||
        offset.y + extent.height > extent_.height ||
        offset.z + extent.depth > extent_.depth ||
        offset.array_layer_offset + extent.array_layer_count >
            extent_.array_layer_count) {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "extent");
      return {};
    }
    TiImageSlice slice{};
    slice.image = image_;
    slice.extent = extent;
    slice.offset = offset;
    slice.mip_level = mip_level;
    return ImageSlice(runtime_, slice);
  }
  inline ImageSlice slice() const {
    return slice(TiImageOffset{}, extent_, 0);
  }

  constexpr TiImageDimension dimension() const {
    return dimension_;
  }
  constexpr const TiImageExtent &extent() const {
    return extent_;
  }
  constexpr uint32_t mip_level_count() const {
    return mip_level_count_;
  }
  constexpr TiFormat format() const {
    return format_;
  }
  constexpr TiImage image() const {
    return image_;
  }
  constexpr operator TiImage() const {  // NOLINT
    return image_;
  }
};

class Texture {
 protected:
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
    if (texture.image != image_.image()) {
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

  inline Texture borrow() const {
    return Texture(image_.borrow(), texture_);
  }

  inline void copy_to(const Texture &dst) const {
    slice().copy_to(dst.slice());
  }

  inline ImageSlice slice() const {
    return image_.slice();
  }

  constexpr const Image &image() const {
    return image_;
  }
  constexpr TiTexture texture() const {
    return texture_;
  }
  constexpr operator TiTexture() const {  // NOLINT
    return texture_;
  }
};

template <typename T>
struct DataTypeToEnum {
  static constexpr TiDataType value = TI_DATA_TYPE_UNKNOWN;
};
#define DEFINE_DATA_TYPE_ENUM(type, enumv)                    \
  template <>                                                 \
  struct DataTypeToEnum<type> {                               \
    static constexpr TiDataType value = TI_DATA_TYPE_##enumv; \
  };

DEFINE_DATA_TYPE_ENUM(int32_t, I32);
DEFINE_DATA_TYPE_ENUM(float, F32);
DEFINE_DATA_TYPE_ENUM(uint16_t, U16);
DEFINE_DATA_TYPE_ENUM(int16_t, I16);
DEFINE_DATA_TYPE_ENUM(uint8_t, U8);
DEFINE_DATA_TYPE_ENUM(int8_t, I8);
DEFINE_DATA_TYPE_ENUM(uint64_t, U64);
DEFINE_DATA_TYPE_ENUM(int64_t, I64);
#undef DEFINE_DATA_TYPE_ENUM

class ArgumentEntry {
  friend class ComputeGraph;
  TiArgument *arg_;

 public:
  ArgumentEntry() = delete;
  ArgumentEntry(const ArgumentEntry &) = delete;
  ArgumentEntry(ArgumentEntry &&b) : arg_(b.arg_) {
  }
  explicit ArgumentEntry(TiArgument *arg) : arg_(arg) {
  }

  inline void set_f16(float value) {
    arg_->type = TI_ARGUMENT_TYPE_SCALAR;
    arg_->value.scalar.type = TI_DATA_TYPE_F16;
    std::memcpy(&arg_->value.scalar.value.x32, &value, sizeof(value));
  }
  inline void set_u16(uint16_t value) {
    arg_->type = TI_ARGUMENT_TYPE_SCALAR;
    arg_->value.scalar.type = TI_DATA_TYPE_U16;
    std::memcpy(&arg_->value.scalar.value.x16, &value, sizeof(value));
  }
  inline void set_i16(int16_t value) {
    arg_->type = TI_ARGUMENT_TYPE_SCALAR;
    arg_->value.scalar.type = TI_DATA_TYPE_I16;
    std::memcpy(&arg_->value.scalar.value.x16, &value, sizeof(value));
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
  inline ArgumentEntry &operator=(uint16_t u16) {
    this->set_u16(u16);
    return *this;
  }
  inline ArgumentEntry &operator=(int16_t i16) {
    this->set_i16(i16);
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
  template <typename T>
  inline ArgumentEntry &operator=(const std::vector<T> &matrix) {
    arg_->type = TI_ARGUMENT_TYPE_TENSOR;
    std::memcpy(arg_->value.tensor.contents.data.x8, matrix.data(),
                matrix.size() * sizeof(T));
    arg_->value.tensor.contents.length = matrix.size();
    arg_->value.tensor.type = DataTypeToEnum<T>::value;
    return *this;
  }
  template <typename T>
  inline ArgumentEntry &operator=(const std::vector<std::vector<T>> &matrix) {
    arg_->type = TI_ARGUMENT_TYPE_TENSOR;
    uint32_t size = 0, bias = 0;
    for (const auto &row : matrix) {
      std::memcpy((arg_->value.tensor.contents.data.x8 + bias), row.data(),
                  row.size() * sizeof(T));
      size += row.size();
      bias += row.size() * sizeof(T);
    }
    arg_->value.tensor.contents.length = size;
    arg_->value.tensor.type = DataTypeToEnum<T>::value;
    return *this;
  }
};

class ComputeGraph {
 protected:
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

  void launch(uint32_t argument_count, const TiNamedArgument *arguments) const {
    ti_launch_compute_graph(runtime_, compute_graph_, argument_count,
                            arguments);
  }
  void launch() const {
    launch(args_.size(), args_.data());
  }
  void launch(const std::vector<TiNamedArgument> &arguments) const {
    launch(arguments.size(), arguments.data());
  }

  constexpr TiComputeGraph compute_graph() const {
    return compute_graph_;
  }
  constexpr operator TiComputeGraph() const {  // NOLINT
    return compute_graph_;
  }
};

class Kernel {
 protected:
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
    args_.resize(idx + 1);
    args_[idx].type = TI_ARGUMENT_TYPE_TENSOR;
    std::memcpy(args_[idx].value.tensor.contents.data.x32, v.data(),
                v.size() * sizeof(T));
    args_[idx].value.tensor.contents.length = v.size();
    args_[idx].value.tensor.type = DataTypeToEnum<T>::value;
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

  void launch(uint32_t argument_count, const TiArgument *arguments) const {
    ti_launch_kernel(runtime_, kernel_, argument_count, arguments);
  }
  void launch() const {
    launch(args_.size(), args_.data());
  }
  void launch(const std::vector<TiArgument> &arguments) const {
    launch(arguments.size(), arguments.data());
  }

  constexpr TiKernel kernel() const {
    return kernel_;
  }
  constexpr operator TiKernel() const {  // NOLINT
    return kernel_;
  }
};

class AotModule {
 protected:
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
        should_destroy_(detail::exchange(b.should_destroy_, false)) {
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
    should_destroy_ = detail::exchange(b.should_destroy_, false);
    return *this;
  }

  inline AotModule borrow() const {
    return AotModule(runtime_, aot_module_, false);
  }

  Kernel get_kernel(const char *name) const {
    TiKernel kernel_ = ti_get_aot_module_kernel(aot_module_, name);
    return Kernel(runtime_, kernel_);
  }
  ComputeGraph get_compute_graph(const char *name) const {
    TiComputeGraph compute_graph_ =
        ti_get_aot_module_compute_graph(aot_module_, name);
    return ComputeGraph(runtime_, compute_graph_);
  }

  constexpr TiAotModule aot_module() const {
    return aot_module_;
  }
  constexpr operator TiAotModule() const {  // NOLINT
    return aot_module_;
  }
};

class CapabilityLevelConfigBuilder;
class CapabilityLevelConfig {
 public:
  std::vector<TiCapabilityLevelInfo> cap_level_infos;

  CapabilityLevelConfig() {
  }
  explicit CapabilityLevelConfig(
      std::vector<TiCapabilityLevelInfo> &&capabilities)
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
  std::map<TiCapability, uint32_t> cap_level_infos_;

 public:
  CapabilityLevelConfigBuilder() {
  }
  CapabilityLevelConfigBuilder(const Self &) = delete;
  Self &operator=(const Self &) = delete;

  Self &spirv_version(uint32_t major, uint32_t minor) {
    if (major == 1) {
      if (minor == 3) {
        cap_level_infos_[TI_CAPABILITY_SPIRV_VERSION] = 0x10300;
      } else if (minor == 4) {
        cap_level_infos_[TI_CAPABILITY_SPIRV_VERSION] = 0x10400;
      } else if (minor == 5) {
        cap_level_infos_[TI_CAPABILITY_SPIRV_VERSION] = 0x10500;
      } else {
        ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "minor");
      }
    } else {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "major");
    }
    return *this;
  }
  Self &spirv_has_int8(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_INT8] = value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_int16(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_INT16] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_int64(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_INT64] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_float16(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_FLOAT16] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_float64(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_FLOAT64] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_int64(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_ATOMIC_INT64] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float16(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float16_add(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16_ADD] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float16_minmax(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT16_MINMAX] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float64(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_atomic_float64_add(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_ATOMIC_FLOAT64_ADD] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_variable_ptr(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_VARIABLE_PTR] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_physical_storage_buffer(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_PHYSICAL_STORAGE_BUFFER] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_subgroup_basic(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_SUBGROUP_BASIC] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_subgroup_vote(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_SUBGROUP_VOTE] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_subgroup_arithmetic(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_SUBGROUP_ARITHMETIC] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_subgroup_ballot(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_SUBGROUP_BALLOT] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_non_semantic_info(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_NON_SEMANTIC_INFO] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }
  Self &spirv_has_no_integer_wrap_decoration(bool value = true) {
    cap_level_infos_[TI_CAPABILITY_SPIRV_HAS_NO_INTEGER_WRAP_DECORATION] =
        value ? TI_TRUE : TI_FALSE;
    return *this;
  }

  CapabilityLevelConfig build() {
    std::vector<TiCapabilityLevelInfo> out{};
    for (const auto &pair : cap_level_infos_) {
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
 protected:
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
      : arch_(detail::exchange(b.arch_, TI_ARCH_MAX_ENUM)),
        runtime_(detail::move_handle(b.runtime_)),
        should_destroy_(detail::exchange(b.should_destroy_, false)) {
  }
  explicit Runtime(TiArch arch, uint32_t device_index = 0)
      : arch_(arch),
        runtime_(ti_create_runtime(arch, device_index)),
        should_destroy_(true) {
  }
  Runtime(TiArch arch, TiRuntime runtime, bool should_destroy)
      : arch_(arch), runtime_(runtime), should_destroy_(should_destroy) {
  }
  ~Runtime() {
    destroy();
  }

  Runtime &operator=(const Runtime &) = delete;
  Runtime &operator=(Runtime &&b) {
    arch_ = detail::exchange(b.arch_, TI_ARCH_MAX_ENUM);
    runtime_ = detail::move_handle(b.runtime_);
    should_destroy_ = detail::exchange(b.should_destroy_, false);
    return *this;
  }

  inline Runtime borrow() const {
    return Runtime(arch_, runtime_, false);
  }

  void set_capabilities_ext(
      const std::vector<TiCapabilityLevelInfo> &capabilities) const {
    ti_set_runtime_capabilities_ext(runtime_, (uint32_t)capabilities.size(),
                                    capabilities.data());
  }
  void set_capabilities_ext(const CapabilityLevelConfig &capabilities) const {
    set_capabilities_ext(capabilities.cap_level_infos);
  }
  CapabilityLevelConfig get_capabilities() const {
    uint32_t n = 0;
    ti_get_runtime_capabilities(runtime_, &n, nullptr);
    std::vector<TiCapabilityLevelInfo> devcaps(n);
    ti_get_runtime_capabilities(runtime_, &n, devcaps.data());
    return CapabilityLevelConfig{std::move(devcaps)};
  }

  Memory allocate_memory(const TiMemoryAllocateInfo &allocate_info) const {
    TiMemory memory = ti_allocate_memory(runtime_, &allocate_info);
    return Memory(runtime_, memory, allocate_info.size, true);
  }
  Memory allocate_memory(size_t size, bool host_access = false) const {
    TiMemoryAllocateInfo allocate_info{};
    allocate_info.size = size;
    allocate_info.host_read = host_access;
    allocate_info.host_write = host_access;
    allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    return allocate_memory(allocate_info);
  }
  template <typename T>
  NdArray<T> allocate_ndarray(const std::vector<uint32_t> &shape = {},
                              const std::vector<uint32_t> &elem_shape = {},
                              bool host_access = false) const {
    auto dtype = detail::templ2dtype<T>::value;
    return allocate_ndarray<T>(dtype, shape, elem_shape, host_access);
  }

  template <typename T>
  NdArray<T> allocate_ndarray(TiDataType dtype,
                              const std::vector<uint32_t> &shape = {},
                              const std::vector<uint32_t> &elem_shape = {},
                              bool host_access = false) const {
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
    ndarray.elem_type = dtype;

    ti::Memory memory = allocate_memory(size, host_access);
    ndarray.memory = memory.memory();
    return NdArray<T>(std::move(memory), ndarray);
  }

  Image allocate_image(const TiImageAllocateInfo &allocate_info) const {
    TiImage image = ti_allocate_image(runtime_, &allocate_info);
    return Image(runtime_, image, allocate_info.dimension, allocate_info.extent,
                 allocate_info.mip_level_count, allocate_info.format, true);
  }
  Texture allocate_texture2d(uint32_t width,
                             uint32_t height,
                             TiFormat format,
                             TiSampler sampler) const {
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
    texture.image = image.image();
    texture.dimension = TI_IMAGE_DIMENSION_2D;
    texture.extent = extent;
    texture.format = format;
    texture.sampler = sampler;
    return Texture(std::move(image), texture);
  }

  AotModule load_aot_module(const char *path) const {
    TiAotModule aot_module_ = ti_load_aot_module(runtime_, path);
    return AotModule(runtime_, aot_module_, true);
  }
  AotModule load_aot_module(const std::string &path) const {
    return load_aot_module(path.c_str());
  }

  AotModule create_aot_module(const void *tcm, size_t size) const {
    TiAotModule aot_module = ti_create_aot_module(runtime_, tcm, size);
    return AotModule(runtime_, aot_module, true);
  }
  AotModule create_aot_module(const std::vector<uint8_t> &tcm) const {
    return create_aot_module(tcm.data(), tcm.size());
  }

  void copy_memory_device_to_device(const MemorySlice &dst_memory,
                                    const MemorySlice &src_memory) const {
    ti_copy_memory_device_to_device(runtime_, &dst_memory.slice(),
                                    &src_memory.slice());
  }
  void copy_image_device_to_device(const ImageSlice &dst_image,
                                   const ImageSlice &src_image) const {
    ti_copy_image_device_to_device(runtime_, &dst_image.slice(),
                                   &src_image.slice());
  }
  void transition_image(TiImage image, TiImageLayout layout) const {
    ti_transition_image(runtime_, image, layout);
  }

  void flush() const {
    ti_flush(runtime_);
  }
  void wait() const {
    ti_wait(runtime_);
  }

  constexpr TiArch arch() const {
    return arch_;
  }
  constexpr TiRuntime runtime() const {
    return runtime_;
  }
  constexpr operator TiRuntime() const {  // NOLINT
    return runtime_;
  }
};

}  // namespace ti
