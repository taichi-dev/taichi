#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/rhi/dx/dx_device.h"

namespace taichi {
namespace lang {

namespace spirv {

using cap = DeviceCapability;

void IRBuilder::init_header() {
  TI_ASSERT(header_.size() == 0U);
  header_.push_back(spv::MagicNumber);

  header_.push_back(device_->get_cap(cap::spirv_version));

  TI_TRACE("SPIR-V Version {}", device_->get_cap(cap::spirv_version));

  // generator: set to 0, unknown
  header_.push_back(0U);
  // Bound: set during Finalize
  header_.push_back(0U);
  // Schema: reserved
  header_.push_back(0U);

  // capability
  ib_.begin(spv::OpCapability).add(spv::CapabilityShader).commit(&header_);

  if (device_->get_cap(cap::spirv_has_atomic_float64_add)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityAtomicFloat64AddEXT)
        .commit(&header_);
  }

  if (device_->get_cap(cap::spirv_has_atomic_float_add)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityAtomicFloat32AddEXT)
        .commit(&header_);
  }

  if (device_->get_cap(cap::spirv_has_atomic_float_minmax)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityAtomicFloat32MinMaxEXT)
        .commit(&header_);
  }

  if (device_->get_cap(cap::spirv_has_variable_ptr)) {
    /*
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityVariablePointers)
        .commit(&header_);
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityVariablePointersStorageBuffer)
        .commit(&header_);
        */
  }

  if (device_->get_cap(cap::spirv_has_int8)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt8).commit(&header_);
  }
  if (device_->get_cap(cap::spirv_has_int16)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt16).commit(&header_);
  }
  if (device_->get_cap(cap::spirv_has_int64)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt64).commit(&header_);
  }
  if (device_->get_cap(cap::spirv_has_float16)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityFloat16).commit(&header_);
  }
  if (device_->get_cap(cap::spirv_has_float64)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityFloat64).commit(&header_);
  }
  if (device_->get_cap(cap::spirv_has_physical_storage_buffer)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityPhysicalStorageBufferAddresses)
        .commit(&header_);
  }

  ib_.begin(spv::OpExtension)
      .add("SPV_KHR_storage_buffer_storage_class")
      .commit(&header_);

  if (device_->get_cap(cap::spirv_has_variable_ptr)) {
    ib_.begin(spv::OpExtension)
        .add("SPV_KHR_variable_pointers")
        .commit(&header_);
  }

  if (device_->get_cap(cap::spirv_has_atomic_float_add)) {
    ib_.begin(spv::OpExtension)
        .add("SPV_EXT_shader_atomic_float_add")
        .commit(&header_);
  }

  if (device_->get_cap(cap::spirv_has_atomic_float_minmax)) {
    ib_.begin(spv::OpExtension)
        .add("SPV_EXT_shader_atomic_float_min_max")
        .commit(&header_);
  }

  if (device_->get_cap(cap::spirv_has_physical_storage_buffer)) {
    ib_.begin(spv::OpExtension)
        .add("SPV_KHR_physical_storage_buffer")
        .commit(&header_);

    // memory model
    ib_.begin(spv::OpMemoryModel)
        .add_seq(spv::AddressingModelPhysicalStorageBuffer64,
                 spv::MemoryModelGLSL450)
        .commit(&entry_);
  } else {
    ib_.begin(spv::OpMemoryModel)
        .add_seq(spv::AddressingModelLogical, spv::MemoryModelGLSL450)
        .commit(&entry_);
  }

  this->init_pre_defs();
}

std::vector<uint32_t> IRBuilder::finalize() {
  std::vector<uint32_t> data;
  // set bound
  const int bound_loc = 3;
  header_[bound_loc] = id_counter_;
  data.insert(data.end(), header_.begin(), header_.end());
  data.insert(data.end(), entry_.begin(), entry_.end());
  data.insert(data.end(), exec_mode_.begin(), exec_mode_.end());
  data.insert(data.end(), debug_.begin(), debug_.end());
  data.insert(data.end(), decorate_.begin(), decorate_.end());
  data.insert(data.end(), global_.begin(), global_.end());
  data.insert(data.end(), func_header_.begin(), func_header_.end());
  data.insert(data.end(), function_.begin(), function_.end());
  return data;
}

void IRBuilder::init_pre_defs() {
  ext_glsl450_ = ext_inst_import("GLSL.std.450");
  t_bool_ = declare_primitive_type(get_data_type<bool>());
  if (device_->get_cap(cap::spirv_has_int8)) {
    t_int8_ = declare_primitive_type(get_data_type<int8>());
    t_uint8_ = declare_primitive_type(get_data_type<uint8>());
  }
  if (device_->get_cap(cap::spirv_has_int16)) {
    t_int16_ = declare_primitive_type(get_data_type<int16>());
    t_uint16_ = declare_primitive_type(get_data_type<uint16>());
  }
  t_int32_ = declare_primitive_type(get_data_type<int32>());
  t_uint32_ = declare_primitive_type(get_data_type<uint32>());
  if (device_->get_cap(cap::spirv_has_int64)) {
    t_int64_ = declare_primitive_type(get_data_type<int64>());
    t_uint64_ = declare_primitive_type(get_data_type<uint64>());
  }
  t_fp32_ = declare_primitive_type(get_data_type<float32>());
  if (device_->get_cap(cap::spirv_has_float16)) {
    t_fp16_ = declare_primitive_type(PrimitiveType::f16);
  }
  if (device_->get_cap(cap::spirv_has_float64)) {
    t_fp64_ = declare_primitive_type(get_data_type<float64>());
  }
  // declare void, and void functions
  t_void_.id = id_counter_++;
  ib_.begin(spv::OpTypeVoid).add(t_void_).commit(&global_);
  t_void_func_.id = id_counter_++;
  ib_.begin(spv::OpTypeFunction)
      .add_seq(t_void_func_, t_void_)
      .commit(&global_);

  // compute shader related types
  t_v2_int_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector)
      .add(t_v2_int_)
      .add_seq(t_int32_, 2)
      .commit(&global_);

  t_v3_int_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector)
      .add(t_v3_int_)
      .add_seq(t_int32_, 3)
      .commit(&global_);

  t_v3_uint_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector)
      .add(t_v3_uint_)
      .add_seq(t_uint32_, 3)
      .commit(&global_);

  t_v4_fp32_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector)
      .add(t_v4_fp32_)
      .add_seq(t_fp32_, 4)
      .commit(&global_);

  t_v2_fp32_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector)
      .add(t_v2_fp32_)
      .add_seq(t_fp32_, 2)
      .commit(&global_);

  t_v3_fp32_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector)
      .add(t_v3_fp32_)
      .add_seq(t_fp32_, 3)
      .commit(&global_);

  // pre-defined constants
  const_i32_zero_ = int_immediate_number(t_int32_, 0);
  const_i32_one_ = int_immediate_number(t_int32_, 1);
}

PhiValue IRBuilder::make_phi(const SType &out_type, uint32_t num_incoming) {
  Value val = new_value(out_type, ValueKind::kNormal);
  ib_.begin(spv::OpPhi).add_seq(out_type, val);
  for (uint32_t i = 0; i < 2 * num_incoming; ++i) {
    ib_.add(0);
  }

  PhiValue phi;
  phi.id = val.id;
  phi.stype = out_type;
  phi.flag = ValueKind::kNormal;
  phi.instr = ib_.commit(&function_);
  return phi;
}

Value IRBuilder::int_immediate_number(const SType &dtype,
                                      int64_t value,
                                      bool cache) {
  return get_const(dtype, reinterpret_cast<uint64_t *>(&value), cache);
}

Value IRBuilder::uint_immediate_number(const SType &dtype,
                                       uint64_t value,
                                       bool cache) {
  return get_const(dtype, &value, cache);
}

Value IRBuilder::float_immediate_number(const SType &dtype,
                                        double value,
                                        bool cache) {
  if (data_type_bits(dtype.dt) == 64) {
    return get_const(dtype, reinterpret_cast<uint64_t *>(&value), cache);
  } else if (data_type_bits(dtype.dt) == 32) {
    float fvalue = static_cast<float>(value);
    uint32_t *ptr = reinterpret_cast<uint32_t *>(&fvalue);
    uint64_t data = ptr[0];
    return get_const(dtype, &data, cache);
  } else if (data_type_bits(dtype.dt) == 16) {
    float fvalue = static_cast<float>(value);
    uint16_t *ptr = reinterpret_cast<uint16_t *>(&fvalue);
    uint64_t data = ptr[0];
    return get_const(dtype, &data, cache);
  } else {
    TI_ERROR("Type {} not supported.", dtype.dt->to_string());
  }
}

SType IRBuilder::get_null_type() {
  SType res;
  res.id = id_counter_++;
  return res;
}

SType IRBuilder::get_primitive_type(const DataType &dt) const {
  if (dt->is_primitive(PrimitiveTypeID::u1)) {
    return t_bool_;
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    if (!device_->get_cap(cap::spirv_has_float16))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_fp16_;
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return t_fp32_;
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    if (!device_->get_cap(cap::spirv_has_float64))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_fp64_;
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    if (!device_->get_cap(cap::spirv_has_int8))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_int8_;
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    if (!device_->get_cap(cap::spirv_has_int16))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_int16_;
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return t_int32_;
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    if (!device_->get_cap(cap::spirv_has_int64))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_int64_;
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    if (!device_->get_cap(cap::spirv_has_int8))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_uint8_;
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    if (!device_->get_cap(cap::spirv_has_int16))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_uint16_;
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return t_uint32_;
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    if (!device_->get_cap(cap::spirv_has_int64))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_uint64_;
  } else {
    TI_ERROR("Type {} not supported.", dt->to_string());
  }
}

size_t IRBuilder::get_primitive_type_size(const DataType &dt) const {
  if (dt == PrimitiveType::i64 || dt == PrimitiveType::u64 ||
      dt == PrimitiveType::f64) {
    return 8;
  } else if (dt == PrimitiveType::i32 || dt == PrimitiveType::u32 ||
             dt == PrimitiveType::f32) {
    return 4;
  } else if (dt == PrimitiveType::i16 || dt == PrimitiveType::u16 ||
             dt == PrimitiveType::f16) {
    return 2;
  } else {
    return 1;
  }
}

SType IRBuilder::get_primitive_uint_type(const DataType &dt) const {
  if (dt == PrimitiveType::i64 || dt == PrimitiveType::u64 ||
      dt == PrimitiveType::f64) {
    return t_uint64_;
  } else if (dt == PrimitiveType::i32 || dt == PrimitiveType::u32 ||
             dt == PrimitiveType::f32) {
    return t_uint32_;
  } else if (dt == PrimitiveType::i16 || dt == PrimitiveType::u16 ||
             dt == PrimitiveType::f16) {
    return t_uint16_;
  } else {
    return t_uint8_;
  }
}

DataType IRBuilder::get_taichi_uint_type(const DataType &dt) const {
  if (dt == PrimitiveType::i64 || dt == PrimitiveType::u64 ||
      dt == PrimitiveType::f64) {
    return PrimitiveType::u64;
  } else if (dt == PrimitiveType::i32 || dt == PrimitiveType::u32 ||
             dt == PrimitiveType::f32) {
    return PrimitiveType::u32;
  } else if (dt == PrimitiveType::i16 || dt == PrimitiveType::u16 ||
             dt == PrimitiveType::f16) {
    return PrimitiveType::u16;
  } else {
    return PrimitiveType::u8;
  }
}

SType IRBuilder::get_pointer_type(const SType &value_type,
                                  spv::StorageClass storage_class) {
  auto key = std::make_pair(value_type.id, storage_class);
  auto it = pointer_type_tbl_.find(key);
  if (it != pointer_type_tbl_.end()) {
    return it->second;
  }
  SType t;
  t.id = id_counter_++;
  t.flag = TypeKind::kPtr;
  t.element_type_id = value_type.id;
  t.storage_class = storage_class;
  ib_.begin(spv::OpTypePointer)
      .add_seq(t, storage_class, value_type)
      .commit(&global_);
  pointer_type_tbl_[key] = t;
  return t;
}

SType IRBuilder::get_sampled_image_type(const SType &primitive_type,
                                        int num_dimensions) {
  auto key = std::make_pair(primitive_type.id, num_dimensions);
  auto it = sampled_image_ptr_tbl_.find(key);
  if (it != sampled_image_ptr_tbl_.end()) {
    return it->second;
  }
  int img_id = id_counter_++;
  spv::Dim dim;
  if (num_dimensions == 1) {
    dim = spv::Dim1D;
  } else if (num_dimensions == 2) {
    dim = spv::Dim2D;
  } else if (num_dimensions == 3) {
    dim = spv::Dim3D;
  } else {
    TI_ERROR("Unsupported number of dimensions: {}", num_dimensions);
  }
  ib_.begin(spv::OpTypeImage)
      .add_seq(img_id, primitive_type, dim,
               /*Depth=*/0, /*Arrayed=*/0, /*MS=*/0, /*Sampled=*/1,
               spv::ImageFormatUnknown)
      .commit(&global_);
  SType sampled_t;
  sampled_t.id = id_counter_++;
  sampled_t.flag = TypeKind::kImage;
  ib_.begin(spv::OpTypeSampledImage)
      .add_seq(sampled_t, img_id)
      .commit(&global_);
  sampled_image_ptr_tbl_[key] = sampled_t;
  return sampled_t;
}

SType IRBuilder::get_storage_image_type(BufferFormat format,
                                        int num_dimensions) {
  auto key = std::make_pair(format, num_dimensions);
  auto it = storage_image_ptr_tbl_.find(key);
  if (it != storage_image_ptr_tbl_.end()) {
    return it->second;
  }
  int img_id = id_counter_++;

  spv::Dim dim;
  if (num_dimensions == 1) {
    dim = spv::Dim1D;
  } else if (num_dimensions == 2) {
    dim = spv::Dim2D;
  } else if (num_dimensions == 3) {
    dim = spv::Dim3D;
  } else {
    TI_ERROR("Unsupported number of dimensions: {}", num_dimensions);
  }

  const std::unordered_map<BufferFormat, spv::ImageFormat> format2spv = {
      {BufferFormat::r8, spv::ImageFormatR8},
      {BufferFormat::rg8, spv::ImageFormatRg8},
      {BufferFormat::rgba8, spv::ImageFormatRgba8},
      {BufferFormat::rgba8srgb, spv::ImageFormatRgba8},
      {BufferFormat::r8u, spv::ImageFormatR8ui},
      {BufferFormat::rg8u, spv::ImageFormatRg8ui},
      {BufferFormat::rgba8u, spv::ImageFormatRgba8ui},
      {BufferFormat::r8i, spv::ImageFormatR8i},
      {BufferFormat::rg8i, spv::ImageFormatRg8i},
      {BufferFormat::rgba8i, spv::ImageFormatRgba8i},
      {BufferFormat::r16, spv::ImageFormatR16},
      {BufferFormat::rg16, spv::ImageFormatRg16},
      {BufferFormat::rgba16, spv::ImageFormatRgba16},
      {BufferFormat::r16u, spv::ImageFormatR16ui},
      {BufferFormat::rg16u, spv::ImageFormatRg16ui},
      {BufferFormat::rgba16u, spv::ImageFormatRgba16ui},
      {BufferFormat::r16i, spv::ImageFormatR16i},
      {BufferFormat::rg16i, spv::ImageFormatRg16i},
      {BufferFormat::rgba16i, spv::ImageFormatRgba16i},
      {BufferFormat::r16f, spv::ImageFormatR16f},
      {BufferFormat::rg16f, spv::ImageFormatRg16f},
      {BufferFormat::rgba16f, spv::ImageFormatRgba16f},
      {BufferFormat::r32u, spv::ImageFormatR32ui},
      {BufferFormat::rg32u, spv::ImageFormatRg32ui},
      {BufferFormat::rgba32u, spv::ImageFormatRgba32ui},
      {BufferFormat::r32i, spv::ImageFormatR32i},
      {BufferFormat::rg32i, spv::ImageFormatRg32i},
      {BufferFormat::rgba32i, spv::ImageFormatRgba32i},
      {BufferFormat::r32f, spv::ImageFormatR32f},
      {BufferFormat::rg32f, spv::ImageFormatRg32f},
      {BufferFormat::rgba32f, spv::ImageFormatRgba32f},
      {BufferFormat::depth16, spv::ImageFormatR16},
      {BufferFormat::depth32f, spv::ImageFormatR32f}};

  if (format2spv.find(format) == format2spv.end()) {
    TI_ERROR("Unsupported image format", num_dimensions);
  }
  spv::ImageFormat spv_format = format2spv.at(format);

  // TODO: Add integer type support
  ib_.begin(spv::OpTypeImage)
      .add_seq(img_id, f32_type(), dim,
               /*Depth=*/0, /*Arrayed=*/0, /*MS=*/0, /*Sampled=*/2, spv_format)
      .commit(&global_);
  SType img_t;
  img_t.id = img_id;
  img_t.flag = TypeKind::kImage;
  storage_image_ptr_tbl_[key] = img_t;
  return img_t;
}

SType IRBuilder::get_storage_pointer_type(const SType &value_type) {
  spv::StorageClass storage_class;
  if (device_->get_cap(cap::spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  return get_pointer_type(value_type, storage_class);
}

SType IRBuilder::get_array_type(const SType &value_type, uint32_t num_elems) {
  SType arr_type;
  arr_type.id = id_counter_++;
  arr_type.flag = TypeKind::kPtr;
  arr_type.element_type_id = value_type.id;

  if (num_elems != 0) {
    Value length = uint_immediate_number(
        get_primitive_type(get_data_type<uint32>()), num_elems);
    ib_.begin(spv::OpTypeArray)
        .add_seq(arr_type, value_type, length)
        .commit(&global_);
  } else {
    ib_.begin(spv::OpTypeRuntimeArray)
        .add_seq(arr_type, value_type)
        .commit(&global_);
  }

  uint32_t nbytes;
  if (value_type.flag == TypeKind::kPrimitive) {
    const auto nbits = data_type_bits(value_type.dt);
    nbytes = static_cast<uint32_t>(nbits) / 8;
  } else if (value_type.flag == TypeKind::kSNodeStruct) {
    nbytes = value_type.snode_desc.container_stride;
  } else {
    TI_ERROR("buffer type must be primitive or snode struct");
  }

  if (nbytes == 0) {
    if (value_type.flag == TypeKind::kPrimitive) {
      TI_WARN("Invalid primitive bit size");
    } else {
      TI_WARN("Invalid container stride");
    }
  }

  // decorate the array type
  this->decorate(spv::OpDecorate, arr_type, spv::DecorationArrayStride, nbytes);

  return arr_type;
}

SType IRBuilder::get_struct_array_type(const SType &value_type,
                                       uint32_t num_elems) {
  SType arr_type = get_array_type(value_type, num_elems);

  // declare struct of array
  SType struct_type;
  struct_type.id = id_counter_++;
  struct_type.flag = TypeKind::kStruct;
  struct_type.element_type_id = value_type.id;
  ib_.begin(spv::OpTypeStruct).add_seq(struct_type, arr_type).commit(&global_);
  // decorate the array type.
  ib_.begin(spv::OpMemberDecorate)
      .add_seq(struct_type, 0, spv::DecorationOffset, 0)
      .commit(&decorate_);

  if (device_->get_cap(cap::spirv_version) < 0x10300) {
    // NOTE: BufferBlock was deprecated in SPIRV 1.3
    // use StorageClassStorageBuffer instead.
    // runtime array are always decorated as BufferBlock(shader storage buffer)
    if (num_elems == 0) {
      this->decorate(spv::OpDecorate, struct_type, spv::DecorationBufferBlock);
    }
  } else {
    this->decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);
  }

  return struct_type;
}

SType IRBuilder::create_struct_type(
    std::vector<std::tuple<SType, std::string, size_t>> &components) {
  SType struct_type;
  struct_type.id = id_counter_++;
  struct_type.flag = TypeKind::kStruct;

  auto &builder = ib_.begin(spv::OpTypeStruct).add_seq(struct_type);

  for (auto &[type, name, offset] : components) {
    builder.add_seq(type);
  }

  builder.commit(&global_);

  int i = 0;
  for (auto &[type, name, offset] : components) {
    this->decorate(spv::OpMemberDecorate, struct_type, i, spv::DecorationOffset,
                   offset);
    this->debug(spv::OpMemberName, struct_type, i, name);
    i++;
  }

  return struct_type;
}

Value IRBuilder::buffer_struct_argument(const SType &struct_type,
                                        uint32_t descriptor_set,
                                        uint32_t binding,
                                        const std::string &name) {
  // NOTE: BufferBlock was deprecated in SPIRV 1.3
  // use StorageClassStorageBuffer instead.
  spv::StorageClass storage_class;
  if (device_->get_cap(cap::spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  this->debug(spv::OpName, struct_type, name + "_t");

  if (device_->get_cap(cap::spirv_version) < 0x10300) {
    // NOTE: BufferBlock was deprecated in SPIRV 1.3
    // use StorageClassStorageBuffer instead.
    // runtime array are always decorated as BufferBlock(shader storage buffer)
    this->decorate(spv::OpDecorate, struct_type, spv::DecorationBufferBlock);
  } else {
    this->decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);
  }

  SType ptr_type = get_pointer_type(struct_type, storage_class);

  this->debug(spv::OpName, ptr_type, name + "_ptr");

  Value val = new_value(ptr_type, ValueKind::kStructArrayPtr);
  ib_.begin(spv::OpVariable)
      .add_seq(ptr_type, val, storage_class)
      .commit(&global_);

  this->debug(spv::OpName, val, name);

  this->decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet,
                 descriptor_set);
  this->decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);
  return val;
}

Value IRBuilder::uniform_struct_argument(const SType &struct_type,
                                         uint32_t descriptor_set,
                                         uint32_t binding,
                                         const std::string &name) {
  // NOTE: BufferBlock was deprecated in SPIRV 1.3
  // use StorageClassStorageBuffer instead.
  spv::StorageClass storage_class = spv::StorageClassUniform;

  this->debug(spv::OpName, struct_type, name + "_t");

  this->decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);

  SType ptr_type = get_pointer_type(struct_type, storage_class);

  this->debug(spv::OpName, ptr_type, name + "_ptr");

  Value val = new_value(ptr_type, ValueKind::kStructArrayPtr);
  ib_.begin(spv::OpVariable)
      .add_seq(ptr_type, val, storage_class)
      .commit(&global_);

  this->debug(spv::OpName, val, name);

  this->decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet,
                 descriptor_set);
  this->decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);
  return val;
}

Value IRBuilder::buffer_argument(const SType &value_type,
                                 uint32_t descriptor_set,
                                 uint32_t binding,
                                 const std::string &name) {
  // NOTE: BufferBlock was deprecated in SPIRV 1.3
  // use StorageClassStorageBuffer instead.
  spv::StorageClass storage_class;
  if (device_->get_cap(cap::spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  SType sarr_type = get_struct_array_type(value_type, 0);

  auto typed_name = name + "_" + value_type.dt.to_string();

  this->debug(spv::OpName, sarr_type, typed_name + "_struct_array");

  SType ptr_type = get_pointer_type(sarr_type, storage_class);

  this->debug(spv::OpName, sarr_type, typed_name + "_ptr");

  Value val = new_value(ptr_type, ValueKind::kStructArrayPtr);
  ib_.begin(spv::OpVariable)
      .add_seq(ptr_type, val, storage_class)
      .commit(&global_);

  this->debug(spv::OpName, val, typed_name);

  this->decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet,
                 descriptor_set);
  this->decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);
  return val;
}

Value IRBuilder::struct_array_access(const SType &res_type,
                                     Value buffer,
                                     Value index) {
  TI_ASSERT(buffer.flag == ValueKind::kStructArrayPtr);
  TI_ASSERT(res_type.flag == TypeKind::kPrimitive);

  spv::StorageClass storage_class;
  if (device_->get_cap(cap::spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  SType ptr_type = this->get_pointer_type(res_type, storage_class);
  Value ret = new_value(ptr_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpAccessChain)
      .add_seq(ptr_type, ret, buffer, const_i32_zero_, index)
      .commit(&function_);

  return ret;
}

Value IRBuilder::texture_argument(int num_channels,
                                  int num_dimensions,
                                  uint32_t descriptor_set,
                                  uint32_t binding) {
  auto texture_type = this->get_sampled_image_type(f32_type(), num_dimensions);
  auto texture_ptr_type =
      get_pointer_type(texture_type, spv::StorageClassUniformConstant);

  Value val = new_value(texture_ptr_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpVariable)
      .add_seq(texture_ptr_type, val, spv::StorageClassUniformConstant)
      .commit(&global_);

  this->decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet,
                 descriptor_set);
  this->decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);

  this->debug(spv::OpName, val, "tex");

  this->global_values.push_back(val);

  return val;
}

Value IRBuilder::storage_image_argument(int num_channels,
                                        int num_dimensions,
                                        uint32_t descriptor_set,
                                        uint32_t binding,
                                        BufferFormat format) {
  auto texture_type = this->get_storage_image_type(format, num_dimensions);
  auto texture_ptr_type =
      get_pointer_type(texture_type, spv::StorageClassUniformConstant);

  Value val = new_value(texture_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpVariable)
      .add_seq(texture_ptr_type, val, spv::StorageClassUniformConstant)
      .commit(&global_);

  this->decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet,
                 descriptor_set);
  this->decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);

  this->debug(spv::OpName, val, "tex");

  this->global_values.push_back(val);

  return val;
}

Value IRBuilder::sample_texture(Value texture_var,
                                const std::vector<Value> &args,
                                Value lod) {
  auto image = this->load_variable(
      texture_var, this->get_sampled_image_type(f32_type(), args.size()));
  Value uv;
  if (args.size() == 1) {
    uv = args[0];
  } else if (args.size() == 2) {
    uv = make_value(spv::OpCompositeConstruct, t_v2_fp32_, args[0], args[1]);
  } else if (args.size() == 3) {
    uv = make_value(spv::OpCompositeConstruct, t_v3_fp32_, args[0], args[1],
                    args[2]);
  } else {
    TI_ERROR("Unsupported number of texture coordinates");
  }
  uint32_t lod_operand = 0x2;
  auto res_vec4 = make_value(spv::OpImageSampleExplicitLod, t_v4_fp32_, image,
                             uv, lod_operand, lod);
  return res_vec4;
}

Value IRBuilder::fetch_texel(Value texture_var,
                             const std::vector<Value> &args,
                             Value lod) {
  auto image = this->load_variable(
      texture_var, this->get_sampled_image_type(f32_type(), args.size()));
  Value uv;
  if (args.size() == 1) {
    uv = args[0];
  } else if (args.size() == 2) {
    uv = make_value(spv::OpCompositeConstruct, t_v2_int_, args[0], args[1]);
  } else if (args.size() == 3) {
    uv = make_value(spv::OpCompositeConstruct, t_v3_int_, args[0], args[1],
                    args[2]);
  } else {
    TI_ERROR("Unsupported number of texture coordinates");
  }
  uint32_t lod_operand = 0x2;
  auto res_vec4 =
      make_value(spv::OpImageFetch, t_v4_fp32_, image, uv, lod_operand, lod);
  return res_vec4;
}

Value IRBuilder::image_load(Value image_var, const std::vector<Value> &args) {
  auto image = this->load_variable(image_var, image_var.stype);
  Value uv;
  if (args.size() == 1) {
    uv = args[0];
  } else if (args.size() == 2) {
    uv = make_value(spv::OpCompositeConstruct, t_v2_int_, args[0], args[1]);
  } else if (args.size() == 3) {
    uv = make_value(spv::OpCompositeConstruct, t_v3_int_, args[0], args[1],
                    args[2]);
  } else {
    TI_ERROR("Unsupported number of texture coordinates");
  }
  auto res_vec4 = make_value(spv::OpImageRead, t_v4_fp32_, image, uv);
  return res_vec4;
}

void IRBuilder::image_store(Value image_var, const std::vector<Value> &args) {
  auto image = this->load_variable(image_var, image_var.stype);
  Value uv;
  if (args.size() == 1 + 4) {
    uv = args[0];
  } else if (args.size() == 2 + 4) {
    uv = make_value(spv::OpCompositeConstruct, t_v2_int_, args[0], args[1]);
  } else if (args.size() == 3 + 4) {
    uv = make_value(spv::OpCompositeConstruct, t_v3_int_, args[0], args[1],
                    args[2]);
  } else {
    TI_ERROR("Unsupported number of image coordinates");
  }
  int base = args.size() - 4;
  Value data = make_value(spv::OpCompositeConstruct, t_v4_fp32_, args[base],
                          args[base + 1], args[base + 2], args[base + 3]);
  make_inst(spv::OpImageWrite, image, uv, data);
}

void IRBuilder::set_work_group_size(const std::array<int, 3> group_size) {
  Value size_x =
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(group_size[0]));
  Value size_y =
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(group_size[1]));
  Value size_z =
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(group_size[2]));

  if (gl_work_group_size_.id == 0) {
    gl_work_group_size_.id = id_counter_++;
  }
  ib_.begin(spv::OpConstantComposite)
      .add_seq(t_v3_uint_, gl_work_group_size_, size_x, size_y, size_z)
      .commit(&global_);
  this->decorate(spv::OpDecorate, gl_work_group_size_, spv::DecorationBuiltIn,
                 spv::BuiltInWorkgroupSize);
}

Value IRBuilder::get_num_work_groups(uint32_t dim_index) {
  if (gl_num_work_groups_.id == 0) {
    SType ptr_type = this->get_pointer_type(t_v3_uint_, spv::StorageClassInput);
    gl_num_work_groups_ = new_value(ptr_type, ValueKind::kVectorPtr);
    ib_.begin(spv::OpVariable)
        .add_seq(ptr_type, gl_num_work_groups_, spv::StorageClassInput)
        .commit(&global_);
    this->decorate(spv::OpDecorate, gl_num_work_groups_, spv::DecorationBuiltIn,
                   spv::BuiltInNumWorkgroups);
  }
  SType pint_type = this->get_pointer_type(t_uint32_, spv::StorageClassInput);
  Value ptr = this->make_value(
      spv::OpAccessChain, pint_type, gl_num_work_groups_,
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(dim_index)));

  return this->make_value(spv::OpLoad, t_uint32_, ptr);
}

Value IRBuilder::get_local_invocation_id(uint32_t dim_index) {
  if (gl_local_invocation_id_.id == 0) {
    SType ptr_type = this->get_pointer_type(t_v3_uint_, spv::StorageClassInput);
    gl_local_invocation_id_ = new_value(ptr_type, ValueKind::kVectorPtr);
    ib_.begin(spv::OpVariable)
        .add_seq(ptr_type, gl_local_invocation_id_, spv::StorageClassInput)
        .commit(&global_);
    this->decorate(spv::OpDecorate, gl_local_invocation_id_,
                   spv::DecorationBuiltIn, spv::BuiltInLocalInvocationId);
  }
  SType pint_type = this->get_pointer_type(t_uint32_, spv::StorageClassInput);
  Value ptr = this->make_value(
      spv::OpAccessChain, pint_type, gl_local_invocation_id_,
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(dim_index)));

  return this->make_value(spv::OpLoad, t_uint32_, ptr);
}

Value IRBuilder::get_global_invocation_id(uint32_t dim_index) {
  if (gl_global_invocation_id_.id == 0) {
    SType ptr_type = this->get_pointer_type(t_v3_uint_, spv::StorageClassInput);
    gl_global_invocation_id_ = new_value(ptr_type, ValueKind::kVectorPtr);
    ib_.begin(spv::OpVariable)
        .add_seq(ptr_type, gl_global_invocation_id_, spv::StorageClassInput)
        .commit(&global_);
    this->decorate(spv::OpDecorate, gl_global_invocation_id_,
                   spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
  }
  SType pint_type = this->get_pointer_type(t_uint32_, spv::StorageClassInput);
  Value ptr = this->make_value(
      spv::OpAccessChain, pint_type, gl_global_invocation_id_,
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(dim_index)));

  return this->make_value(spv::OpLoad, t_uint32_, ptr);
}

Value IRBuilder::get_subgroup_invocation_id() {
  if (subgroup_local_invocation_id_.id == 0) {
    SType ptr_type = this->get_pointer_type(t_uint32_, spv::StorageClassInput);
    subgroup_local_invocation_id_ =
        new_value(ptr_type, ValueKind::kVariablePtr);
    ib_.begin(spv::OpVariable)
        .add_seq(ptr_type, subgroup_local_invocation_id_,
                 spv::StorageClassInput)
        .commit(&global_);
    this->decorate(spv::OpDecorate, subgroup_local_invocation_id_,
                   spv::DecorationBuiltIn,
                   spv::BuiltInSubgroupLocalInvocationId);
    global_values.push_back(subgroup_local_invocation_id_);
  }

  return this->make_value(spv::OpLoad, t_uint32_,
                          subgroup_local_invocation_id_);
}

Value IRBuilder::get_subgroup_size() {
  if (subgroup_size_.id == 0) {
    SType ptr_type = this->get_pointer_type(t_uint32_, spv::StorageClassInput);
    subgroup_size_ = new_value(ptr_type, ValueKind::kVariablePtr);
    ib_.begin(spv::OpVariable)
        .add_seq(ptr_type, subgroup_size_, spv::StorageClassInput)
        .commit(&global_);
    this->decorate(spv::OpDecorate, subgroup_size_, spv::DecorationBuiltIn,
                   spv::BuiltInSubgroupSize);
    global_values.push_back(subgroup_size_);
  }

  return this->make_value(spv::OpLoad, t_uint32_, subgroup_size_);
}

#define DEFINE_BUILDER_BINARY_USIGN_OP(_OpName, _Op)   \
  Value IRBuilder::_OpName(Value a, Value b) {         \
    TI_ASSERT(a.stype.id == b.stype.id);               \
    if (is_integral(a.stype.dt)) {                     \
      return make_value(spv::OpI##_Op, a.stype, a, b); \
    } else {                                           \
      TI_ASSERT(is_real(a.stype.dt));                  \
      return make_value(spv::OpF##_Op, a.stype, a, b); \
    }                                                  \
  }

#define DEFINE_BUILDER_BINARY_SIGN_OP(_OpName, _Op)         \
  Value IRBuilder::_OpName(Value a, Value b) {              \
    TI_ASSERT(a.stype.id == b.stype.id);                    \
    if (is_integral(a.stype.dt) && is_signed(a.stype.dt)) { \
      return make_value(spv::OpS##_Op, a.stype, a, b);      \
    } else if (is_integral(a.stype.dt)) {                   \
      return make_value(spv::OpU##_Op, a.stype, a, b);      \
    } else {                                                \
      TI_ASSERT(is_real(a.stype.dt));                       \
      return make_value(spv::OpF##_Op, a.stype, a, b);      \
    }                                                       \
  }

DEFINE_BUILDER_BINARY_USIGN_OP(add, Add);
DEFINE_BUILDER_BINARY_USIGN_OP(sub, Sub);
DEFINE_BUILDER_BINARY_USIGN_OP(mul, Mul);
DEFINE_BUILDER_BINARY_SIGN_OP(div, Div);

Value IRBuilder::mod(Value a, Value b) {
  TI_ASSERT(a.stype.id == b.stype.id);
  if (is_integral(a.stype.dt) && is_signed(a.stype.dt)) {
    // FIXME: figure out why OpSRem does not work
    return sub(a, mul(b, div(a, b)));
  } else if (is_integral(a.stype.dt)) {
    return make_value(spv::OpUMod, a.stype, a, b);
  } else {
    TI_ASSERT(is_real(a.stype.dt));
    return make_value(spv::OpFRem, a.stype, a, b);
  }
}

#define DEFINE_BUILDER_CMP_OP(_OpName, _Op)                                \
  Value IRBuilder::_OpName(Value a, Value b) {                             \
    TI_ASSERT(a.stype.id == b.stype.id);                                   \
    const auto &bool_type = t_bool_; /* TODO: Only scalar supported now */ \
    if (is_integral(a.stype.dt) && is_signed(a.stype.dt)) {                \
      return make_value(spv::OpS##_Op, bool_type, a, b);                   \
    } else if (is_integral(a.stype.dt)) {                                  \
      return make_value(spv::OpU##_Op, bool_type, a, b);                   \
    } else {                                                               \
      TI_ASSERT(is_real(a.stype.dt));                                      \
      return make_value(spv::OpFOrd##_Op, bool_type, a, b);                \
    }                                                                      \
  }

DEFINE_BUILDER_CMP_OP(lt, LessThan);
DEFINE_BUILDER_CMP_OP(le, LessThanEqual);
DEFINE_BUILDER_CMP_OP(gt, GreaterThan);
DEFINE_BUILDER_CMP_OP(ge, GreaterThanEqual);

#define DEFINE_BUILDER_CMP_UOP(_OpName, _Op)                               \
  Value IRBuilder::_OpName(Value a, Value b) {                             \
    TI_ASSERT(a.stype.id == b.stype.id);                                   \
    const auto &bool_type = t_bool_; /* TODO: Only scalar supported now */ \
    if (is_integral(a.stype.dt)) {                                         \
      return make_value(spv::OpI##_Op, bool_type, a, b);                   \
    } else {                                                               \
      TI_ASSERT(is_real(a.stype.dt));                                      \
      return make_value(spv::OpFOrd##_Op, bool_type, a, b);                \
    }                                                                      \
  }

DEFINE_BUILDER_CMP_UOP(eq, Equal);
DEFINE_BUILDER_CMP_UOP(ne, NotEqual);

Value IRBuilder::select(Value cond, Value a, Value b) {
  TI_ASSERT(a.stype.id == b.stype.id);
  TI_ASSERT(cond.stype.id == t_bool_.id);
  return make_value(spv::OpSelect, a.stype, cond, a, b);
}

Value IRBuilder::cast(const SType &dst_type, Value value) {
  TI_ASSERT(value.stype.id > 0U);
  if (value.stype.id == dst_type.id)
    return value;
  const DataType &from = value.stype.dt;
  const DataType &to = dst_type.dt;
  if (from->is_primitive(PrimitiveTypeID::u1)) {  // Bool
    if (is_integral(to) && is_signed(to)) {       // Bool -> Int
      return select(value, int_immediate_number(dst_type, 1),
                    int_immediate_number(dst_type, 0));
    } else if (is_integral(to) && is_unsigned(to)) {  // Bool -> UInt
      return select(value, uint_immediate_number(dst_type, 1),
                    uint_immediate_number(dst_type, 0));
    } else if (is_real(to)) {  // Bool -> Float
      return make_value(spv::OpConvertUToF, dst_type,
                        select(value, uint_immediate_number(t_uint32_, 1),
                               uint_immediate_number(t_uint32_, 0)));
    } else {
      TI_ERROR("do not support type cast from {} to {}", from.to_string(),
               to.to_string());
      return Value();
    }
  } else if (to->is_primitive(PrimitiveTypeID::u1)) {  // Bool
    if (is_integral(from) && is_signed(from)) {        // Int -> Bool
      return ne(value, int_immediate_number(value.stype, 0));
    } else if (is_integral(from) && is_unsigned(from)) {  // UInt -> Bool
      return ne(value, uint_immediate_number(value.stype, 0));
    } else {
      TI_ERROR("do not support type cast from {} to {}", from.to_string(),
               to.to_string());
      return Value();
    }
  } else if (is_integral(from) && is_integral(to)) {
    auto ret = value;

    if (data_type_bits(from) == data_type_bits(to)) {
      // Same width conversion
      ret = make_value(spv::OpBitcast, dst_type, ret);
    } else {
      // Different width
      // Step 1. Sign extend / truncate value to width of `to`
      // Step 2. Bitcast to signess of `to`
      auto get_signed_type = [](DataType dt) -> DataType {
        // Create a output signed type with the same width as `dt`
        if (data_type_bits(dt) == 8)
          return PrimitiveType::i8;
        else if (data_type_bits(dt) == 16)
          return PrimitiveType::i16;
        else if (data_type_bits(dt) == 32)
          return PrimitiveType::i32;
        else if (data_type_bits(dt) == 64)
          return PrimitiveType::i64;
        else
          return PrimitiveType::unknown;
      };
      auto get_unsigned_type = [](DataType dt) -> DataType {
        // Create a output unsigned type with the same width as `dt`
        if (data_type_bits(dt) == 8)
          return PrimitiveType::u8;
        else if (data_type_bits(dt) == 16)
          return PrimitiveType::u16;
        else if (data_type_bits(dt) == 32)
          return PrimitiveType::u32;
        else if (data_type_bits(dt) == 64)
          return PrimitiveType::u64;
        else
          return PrimitiveType::unknown;
      };

      if (is_signed(from)) {
        ret = make_value(spv::OpSConvert,
                         get_primitive_type(get_signed_type(to)), ret);
      } else {
        ret = make_value(spv::OpUConvert,
                         get_primitive_type(get_unsigned_type(to)), ret);
      }

      ret = make_value(spv::OpBitcast, dst_type, ret);
    }

    return ret;
  } else if (is_real(from) && is_integral(to) &&
             is_signed(to)) {  // Float -> Int
    return make_value(spv::OpConvertFToS, dst_type, value);
  } else if (is_real(from) && is_integral(to) &&
             is_unsigned(to)) {  // Float -> UInt
    return make_value(spv::OpConvertFToU, dst_type, value);
  } else if (is_integral(from) && is_signed(from) &&
             is_real(to)) {  // Int -> Float
    return make_value(spv::OpConvertSToF, dst_type, value);
  } else if (is_integral(from) && is_unsigned(from) &&
             is_real(to)) {  // UInt -> Float
    return make_value(spv::OpConvertUToF, dst_type, value);
  } else if (is_real(from) && is_real(to)) {  // Float -> Float
    return make_value(spv::OpFConvert, dst_type, value);
  } else {
    TI_ERROR("do not support type cast from {} to {}", from.to_string(),
             to.to_string());
    return Value();
  }
}

Value IRBuilder::alloca_variable(const SType &type) {
  SType ptr_type = get_pointer_type(type, spv::StorageClassFunction);
  Value ret = new_value(ptr_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpVariable)
      .add_seq(ptr_type, ret, spv::StorageClassFunction)
      .commit(&func_header_);
  return ret;
}

Value IRBuilder::alloca_workgroup_array(const SType &arr_type) {
  SType ptr_type = get_pointer_type(arr_type, spv::StorageClassWorkgroup);
  Value ret = new_value(ptr_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpVariable)
      .add_seq(ptr_type, ret, spv::StorageClassWorkgroup)
      .commit(&global_);
  return ret;
}

Value IRBuilder::load_variable(Value pointer, const SType &res_type) {
  TI_ASSERT(pointer.flag == ValueKind::kVariablePtr ||
            pointer.flag == ValueKind::kStructArrayPtr ||
            pointer.flag == ValueKind::kPhysicalPtr);
  Value ret = new_value(res_type, ValueKind::kNormal);
  if (pointer.flag == ValueKind::kPhysicalPtr) {
    Value alignment =
        uint_immediate_number(t_uint32_, get_primitive_type_size(res_type.dt));
    ib_.begin(spv::OpLoad)
        .add_seq(res_type, ret, pointer, spv::MemoryAccessAlignedMask,
                 alignment)
        .commit(&function_);
  } else {
    ib_.begin(spv::OpLoad).add_seq(res_type, ret, pointer).commit(&function_);
  }
  return ret;
}
void IRBuilder::store_variable(Value pointer, Value value) {
  TI_ASSERT(pointer.flag == ValueKind::kVariablePtr ||
            pointer.flag == ValueKind::kPhysicalPtr);
  TI_ASSERT(value.stype.id == pointer.stype.element_type_id);
  if (pointer.flag == ValueKind::kPhysicalPtr) {
    Value alignment = uint_immediate_number(
        t_uint32_, get_primitive_type_size(value.stype.dt));
    ib_.begin(spv::OpStore)
        .add_seq(pointer, value, spv::MemoryAccessAlignedMask, alignment)
        .commit(&function_);
  } else {
    ib_.begin(spv::OpStore).add_seq(pointer, value).commit(&function_);
  }
}

void IRBuilder::register_value(std::string name, Value value) {
  auto it = value_name_tbl_.find(name);
  if (it != value_name_tbl_.end() && it->second.flag != ValueKind::kConstant) {
    TI_ERROR("{} already exists.", name);
  }
  this->debug(
      spv::OpName, value,
      fmt::format("{}_{}", name, value.stype.dt.to_string()));  // Debug info
  value_name_tbl_[name] = value;
}

Value IRBuilder::query_value(std::string name) const {
  auto it = value_name_tbl_.find(name);
  if (it != value_name_tbl_.end()) {
    return it->second;
  }
  TI_ERROR("Value \"{}\" does not yet exist.", name);
}

bool IRBuilder::check_value_existence(const std::string &name) const {
  return value_name_tbl_.find(name) != value_name_tbl_.end();
}

Value IRBuilder::float_atomic(AtomicOpType op_type,
                              Value addr_ptr,
                              Value data) {
  auto atomic_func_ = [&](std::function<Value(Value, Value)> atomic_op) {
    Value ret_val_int = alloca_variable(t_uint32_);

    // do-while
    Label head = new_label();
    Label body = new_label();
    Label branch_true = new_label();
    Label branch_false = new_label();
    Label merge = new_label();
    Label exit = new_label();

    make_inst(spv::OpBranch, head);
    start_label(head);
    make_inst(spv::OpLoopMerge, branch_true, merge, 0);
    make_inst(spv::OpBranch, body);
    make_inst(spv::OpLabel, body);
    // while (true)
    {
      // int old = addr_ptr[0];
      Value old_val = load_variable(addr_ptr, t_uint32_);
      // int new = floatBitsToInt(atomic_op(intBitsToFloat(old), data));
      Value old_float = make_value(spv::OpBitcast, t_fp32_, old_val);
      Value new_float = atomic_op(old_float, data);
      Value new_val = make_value(spv::OpBitcast, t_uint32_, new_float);
      // int loaded = atomicCompSwap(vals[0], old, new);
      /*
      * Don't need this part, theoretically
      auto semantics = uint_immediate_number(
          t_uint32_, spv::MemorySemanticsAcquireReleaseMask |
                         spv::MemorySemanticsUniformMemoryMask);
      make_inst(spv::OpMemoryBarrier, const_i32_one_, semantics);
      */
      Value loaded = make_value(
          spv::OpAtomicCompareExchange, t_uint32_, addr_ptr,
          /*scope=*/const_i32_one_, /*semantics if equal=*/const_i32_zero_,
          /*semantics if unequal=*/const_i32_zero_, new_val, old_val);
      // bool ok = (loaded == old);
      Value ok = make_value(spv::OpIEqual, t_bool_, loaded, old_val);
      // int ret_val_int = loaded;
      store_variable(ret_val_int, loaded);
      // if (ok)
      make_inst(spv::OpSelectionMerge, branch_false, 0);
      make_inst(spv::OpBranchConditional, ok, branch_true, branch_false);
      {
        make_inst(spv::OpLabel, branch_true);
        make_inst(spv::OpBranch, exit);
      }
      // else
      {
        make_inst(spv::OpLabel, branch_false);
        make_inst(spv::OpBranch, merge);
      }
      // continue;
      make_inst(spv::OpLabel, merge);
      make_inst(spv::OpBranch, head);
    }
    start_label(exit);

    return make_value(spv::OpBitcast, t_fp32_,
                      load_variable(ret_val_int, t_uint32_));
  };

  if (op_type == AtomicOpType::add) {
    return atomic_func_([&](Value lhs, Value rhs) { return add(lhs, rhs); });
  } else if (op_type == AtomicOpType::sub) {
    return atomic_func_([&](Value lhs, Value rhs) { return sub(lhs, rhs); });
  } else if (op_type == AtomicOpType::min) {
    return atomic_func_([&](Value lhs, Value rhs) {
      return call_glsl450(t_fp32_, /*FMin*/ 37, lhs, rhs);
    });
  } else if (op_type == AtomicOpType::max) {
    return atomic_func_([&](Value lhs, Value rhs) {
      return call_glsl450(t_fp32_, /*FMax*/ 40, lhs, rhs);
    });
  } else {
    TI_NOT_IMPLEMENTED
  }
}

Value IRBuilder::rand_u32(Value global_tmp_) {
  if (!init_rand_) {
    init_random_function(global_tmp_);
  }

  Value _11u = uint_immediate_number(t_uint32_, 11u);
  Value _19u = uint_immediate_number(t_uint32_, 19u);
  Value _8u = uint_immediate_number(t_uint32_, 8u);
  Value _1000000007u = uint_immediate_number(t_uint32_, 1000000007u);
  Value tmp0 = load_variable(rand_x_, t_uint32_);
  Value tmp1 = make_value(spv::OpShiftLeftLogical, t_uint32_, tmp0, _11u);
  Value tmp_t = make_value(spv::OpBitwiseXor, t_uint32_, tmp0, tmp1);  // t
  store_variable(rand_x_, load_variable(rand_y_, t_uint32_));
  store_variable(rand_y_, load_variable(rand_z_, t_uint32_));
  Value tmp_w = load_variable(rand_w_, t_uint32_);  // reuse w
  store_variable(rand_z_, tmp_w);
  Value tmp2 = make_value(spv::OpShiftRightLogical, t_uint32_, tmp_w, _19u);
  Value tmp3 = make_value(spv::OpBitwiseXor, t_uint32_, tmp_w, tmp2);
  Value tmp4 = make_value(spv::OpShiftRightLogical, t_uint32_, tmp_t, _8u);
  Value tmp5 = make_value(spv::OpBitwiseXor, t_uint32_, tmp_t, tmp4);
  Value new_w = make_value(spv::OpBitwiseXor, t_uint32_, tmp3, tmp5);
  store_variable(rand_w_, new_w);
  Value val = make_value(spv::OpIMul, t_uint32_, new_w, _1000000007u);

  return val;
}

Value IRBuilder::rand_f32(Value global_tmp_) {
  if (!init_rand_) {
    init_random_function(global_tmp_);
  }

  Value _1_4294967296f = float_immediate_number(t_fp32_, 1.0f / 4294967296.0f);
  Value tmp0 = rand_u32(global_tmp_);
  Value tmp1 = cast(t_fp32_, tmp0);
  Value val = mul(tmp1, _1_4294967296f);

  return val;
}

Value IRBuilder::rand_i32(Value global_tmp_) {
  if (!init_rand_) {
    init_random_function(global_tmp_);
  }

  Value tmp0 = rand_u32(global_tmp_);
  Value val = cast(t_int32_, tmp0);
  return val;
}

Value IRBuilder::get_const(const SType &dtype,
                           const uint64_t *pvalue,
                           bool cache) {
  auto key = std::make_pair(dtype.id, pvalue[0]);
  if (cache) {
    auto it = const_tbl_.find(key);
    if (it != const_tbl_.end()) {
      return it->second;
    }
  }

  TI_ASSERT(dtype.flag == TypeKind::kPrimitive);
  Value ret = new_value(dtype, ValueKind::kConstant);
  if (dtype.dt->is_primitive(PrimitiveTypeID::u1)) {
    // bool type
    if (*pvalue) {
      ib_.begin(spv::OpConstantTrue).add_seq(dtype, ret);
    } else {
      ib_.begin(spv::OpConstantFalse).add_seq(dtype, ret);
    }
  } else {
    // Integral/floating-point types.
    ib_.begin(spv::OpConstant).add_seq(dtype, ret);
    uint64_t mask = 0xFFFFFFFFUL;
    ib_.add(static_cast<uint32_t>(pvalue[0] & mask));
    if (data_type_bits(dtype.dt) > 32) {
      if (is_integral(dtype.dt)) {
        int64_t sign_mask = 0xFFFFFFFFL;
        const int64_t *sign_ptr = reinterpret_cast<const int64_t *>(pvalue);
        ib_.add(static_cast<uint32_t>((sign_ptr[0] >> 32L) & sign_mask));
      } else {
        ib_.add(static_cast<uint32_t>((pvalue[0] >> 32UL) & mask));
      }
    }
  }

  ib_.commit(&global_);
  if (cache) {
    const_tbl_[key] = ret;
  }
  return ret;
}

SType IRBuilder::declare_primitive_type(DataType dt) {
  SType t;
  t.id = id_counter_++;
  t.dt = dt;
  t.flag = TypeKind::kPrimitive;

  dt.set_is_pointer(false);
  if (dt->is_primitive(PrimitiveTypeID::u1))
    ib_.begin(spv::OpTypeBool).add(t).commit(&global_);
  else if (is_real(dt))
    ib_.begin(spv::OpTypeFloat).add_seq(t, data_type_bits(dt)).commit(&global_);
  else if (is_integral(dt))
    ib_.begin(spv::OpTypeInt)
        .add_seq(t, data_type_bits(dt), static_cast<int>(is_signed(dt)))
        .commit(&global_);
  else {
    TI_ERROR("Type {} not supported.", dt->to_string());
  }

  return t;
}

void IRBuilder::init_random_function(Value global_tmp_) {
  // variables declare
  SType local_type = get_pointer_type(t_uint32_, spv::StorageClassPrivate);
  rand_x_ = new_value(local_type, ValueKind::kVariablePtr);
  rand_y_ = new_value(local_type, ValueKind::kVariablePtr);
  rand_z_ = new_value(local_type, ValueKind::kVariablePtr);
  rand_w_ = new_value(local_type, ValueKind::kVariablePtr);
  global_values.push_back(rand_x_);
  global_values.push_back(rand_y_);
  global_values.push_back(rand_z_);
  global_values.push_back(rand_w_);
  ib_.begin(spv::OpVariable)
      .add_seq(local_type, rand_x_, spv::StorageClassPrivate)
      .commit(&global_);
  ib_.begin(spv::OpVariable)
      .add_seq(local_type, rand_y_, spv::StorageClassPrivate)
      .commit(&global_);
  ib_.begin(spv::OpVariable)
      .add_seq(local_type, rand_z_, spv::StorageClassPrivate)
      .commit(&global_);
  ib_.begin(spv::OpVariable)
      .add_seq(local_type, rand_w_, spv::StorageClassPrivate)
      .commit(&global_);
  debug(spv::OpName, rand_x_, "_rand_x");
  debug(spv::OpName, rand_y_, "_rand_y");
  debug(spv::OpName, rand_z_, "_rand_z");
  debug(spv::OpName, rand_w_, "_rand_w");
  SType gtmp_type = get_pointer_type(t_uint32_, spv::StorageClassStorageBuffer);
  Value rand_gtmp_ = new_value(gtmp_type, ValueKind::kVariablePtr);
  debug(spv::OpName, rand_gtmp_, "rand_gtmp");

  auto load_var = [&](Value pointer, const SType &res_type) {
    TI_ASSERT(pointer.flag == ValueKind::kVariablePtr ||
              pointer.flag == ValueKind::kStructArrayPtr);
    Value ret = new_value(res_type, ValueKind::kNormal);
    ib_.begin(spv::OpLoad)
        .add_seq(res_type, ret, pointer)
        .commit(&func_header_);
    return ret;
  };

  auto store_var = [&](Value pointer, Value value) {
    TI_ASSERT(pointer.flag == ValueKind::kVariablePtr);
    TI_ASSERT(value.stype.id == pointer.stype.element_type_id);
    ib_.begin(spv::OpStore).add_seq(pointer, value).commit(&func_header_);
  };

  // Constant Number
  Value _7654321u = uint_immediate_number(t_uint32_, 7654321u);
  Value _1234567u = uint_immediate_number(t_uint32_, 1234567u);
  Value _9723451u = uint_immediate_number(t_uint32_, 9723451u);
  Value _123456789u = uint_immediate_number(t_uint32_, 123456789u);
  Value _1000000007u = uint_immediate_number(t_uint32_, 1000000007u);
  Value _362436069u = uint_immediate_number(t_uint32_, 362436069u);
  Value _521288629u = uint_immediate_number(t_uint32_, 521288629u);
  Value _88675123u = uint_immediate_number(t_uint32_, 88675123u);
  Value _1 = int_immediate_number(t_uint32_, 1);
  Value _1024 = int_immediate_number(t_uint32_, 1024);

  // init_rand_ segment (inline to main)
  // ad-hoc: hope no kernel will use more than 1024 gtmp variables...
  ib_.begin(spv::OpAccessChain)
      .add_seq(gtmp_type, rand_gtmp_, global_tmp_, const_i32_zero_, _1024)
      .commit(&func_header_);
  // Get gl_GlobalInvocationID.x, assert it has be visited
  // (in generate_serial_kernel/generate_range_for_kernel
  SType pint_type = this->get_pointer_type(t_uint32_, spv::StorageClassInput);
  Value tmp0 = new_value(pint_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpAccessChain)
      .add_seq(pint_type, tmp0, gl_global_invocation_id_,
               uint_immediate_number(t_uint32_, 0))
      .commit(&func_header_);
  Value tmp1 = load_var(tmp0, t_uint32_);
  Value tmp2_ = load_var(rand_gtmp_, t_uint32_);
  Value tmp2 = new_value(t_uint32_, ValueKind::kNormal);
  ib_.begin(spv::OpBitcast)
      .add_seq(t_uint32_, tmp2, tmp2_)
      .commit(&func_header_);
  Value tmp3 = new_value(t_uint32_, ValueKind::kNormal);
  ib_.begin(spv::OpIAdd)
      .add_seq(t_uint32_, tmp3, _7654321u, tmp1)
      .commit(&func_header_);
  Value tmp4 = new_value(t_uint32_, ValueKind::kNormal);
  ib_.begin(spv::OpIMul)
      .add_seq(t_uint32_, tmp4, _9723451u, tmp2)
      .commit(&func_header_);
  Value tmp5 = new_value(t_uint32_, ValueKind::kNormal);
  ib_.begin(spv::OpIAdd)
      .add_seq(t_uint32_, tmp5, _1234567u, tmp4)
      .commit(&func_header_);
  Value tmp6 = new_value(t_uint32_, ValueKind::kNormal);
  ib_.begin(spv::OpIMul)
      .add_seq(t_uint32_, tmp6, tmp3, tmp5)
      .commit(&func_header_);
  Value tmp7 = new_value(t_uint32_, ValueKind::kNormal);
  ib_.begin(spv::OpIMul)
      .add_seq(t_uint32_, tmp7, _123456789u, tmp6)
      .commit(&func_header_);
  Value tmp8 = new_value(t_uint32_, ValueKind::kNormal);
  ib_.begin(spv::OpIMul)
      .add_seq(t_uint32_, tmp8, _1000000007u, tmp7)
      .commit(&func_header_);
  store_var(rand_x_, tmp8);
  store_var(rand_y_, _362436069u);
  store_var(rand_z_, _521288629u);
  store_var(rand_w_, _88675123u);

  // enum spv::Op add_op = spv::OpIAdd;
  bool use_atomic_increment = false;

// use atomic increment for DX API to avoid error X3694
#ifdef TI_WITH_DX11
  if (dynamic_cast<const taichi::lang::directx11::Dx11Device *>(device_)) {
    use_atomic_increment = true;
  }
#endif

  if (use_atomic_increment) {
    Value tmp9 = new_value(t_uint32_, ValueKind::kNormal);
    ib_.begin(spv::Op::OpAtomicIIncrement)
        .add_seq(t_uint32_, tmp9, rand_gtmp_,
                 /*scope_id*/ const_i32_one_,
                 /*semantics*/ const_i32_zero_)
        .commit(&func_header_);
  } else {
    // Yes, this is not an atomic operation, but just fine since no matter
    // how RAND_STATE changes, `gl_GlobalInvocationID.x` can still help
    // us to set different seeds for different threads.
    // Discussion:
    // https://github.com/taichi-dev/taichi/pull/912#discussion_r419021918
    Value tmp9 = load_var(rand_gtmp_, t_uint32_);
    Value tmp10 = new_value(t_uint32_, ValueKind::kNormal);
    ib_.begin(spv::Op::OpIAdd)
        .add_seq(t_uint32_, tmp10, tmp9, _1)
        .commit(&func_header_);
    store_var(rand_gtmp_, tmp10);
  }

  init_rand_ = true;
}

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
