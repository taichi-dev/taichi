#include "taichi/backends/vulkan/spirv_ir_builder.h"

namespace taichi {
namespace lang {
namespace vulkan {

namespace spirv {

using cap = DeviceCapability;

void IRBuilder::init_header() {
  TI_ASSERT(header_.size() == 0U);
  header_.push_back(spv::MagicNumber);

  header_.push_back(device_->get_cap(cap::vk_spirv_version));

  TI_TRACE("SPIR-V Version {}", device_->get_cap(cap::vk_spirv_version));

  // generator: set to 0, unknown
  header_.push_back(0U);
  // Bound: set during Finalize
  header_.push_back(0U);
  // Schema: reserved
  header_.push_back(0U);

  // capability
  ib_.begin(spv::OpCapability).add(spv::CapabilityShader).commit(&header_);

  if (device_->get_cap(cap::vk_has_atomic_float64_add)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityAtomicFloat64AddEXT)
        .commit(&header_);
  }

  if (device_->get_cap(cap::vk_has_atomic_float_add)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityAtomicFloat32AddEXT)
        .commit(&header_);
  }

  if (device_->get_cap(cap::vk_has_atomic_float_minmax)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityAtomicFloat32MinMaxEXT)
        .commit(&header_);
  }

  if (device_->get_cap(cap::vk_has_spv_variable_ptr)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityVariablePointers)
        .commit(&header_);
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityVariablePointersStorageBuffer)
        .commit(&header_);
  }

  if (device_->get_cap(cap::vk_has_int8)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt8).commit(&header_);
  }
  if (device_->get_cap(cap::vk_has_int16)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt16).commit(&header_);
  }
  if (device_->get_cap(cap::vk_has_int64)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt64).commit(&header_);
  }
  if (device_->get_cap(cap::vk_has_float16)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityFloat16).commit(&header_);
  }
  if (device_->get_cap(cap::vk_has_float64)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityFloat64).commit(&header_);
  }

  ib_.begin(spv::OpExtension)
      .add("SPV_KHR_storage_buffer_storage_class")
      .commit(&header_);

  if (device_->get_cap(cap::vk_has_spv_variable_ptr)) {
    ib_.begin(spv::OpExtension)
        .add("SPV_KHR_variable_pointers")
        .commit(&header_);
  }

  if (device_->get_cap(cap::vk_has_atomic_float_add)) {
    ib_.begin(spv::OpExtension)
        .add("SPV_EXT_shader_atomic_float_add")
        .commit(&header_);
  }

  if (device_->get_cap(cap::vk_has_atomic_float_minmax)) {
    ib_.begin(spv::OpExtension)
        .add("SPV_EXT_shader_atomic_float_min_max")
        .commit(&header_);
  }

  // memory model
  ib_.begin(spv::OpMemoryModel)
      .add_seq(spv::AddressingModelLogical, spv::MemoryModelGLSL450)
      .commit(&entry_);

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
  if (any_atomic_) {
    data.insert(data.end(), atomic_functions_.begin(), atomic_functions_.end());
  }
  return data;
}

void IRBuilder::init_pre_defs() {
  ext_glsl450_ = ext_inst_import("GLSL.std.450");
  t_bool_ = declare_primitive_type(get_data_type<bool>());
  if (device_->get_cap(cap::vk_has_int8)) {
    t_int8_ = declare_primitive_type(get_data_type<int8>());
    t_uint8_ = declare_primitive_type(get_data_type<uint8>());
  }
  if (device_->get_cap(cap::vk_has_int16)) {
    t_int16_ = declare_primitive_type(get_data_type<int16>());
    t_uint16_ = declare_primitive_type(get_data_type<uint16>());
  }
  t_int32_ = declare_primitive_type(get_data_type<int32>());
  t_uint32_ = declare_primitive_type(get_data_type<uint32>());
  if (device_->get_cap(cap::vk_has_int64)) {
    t_int64_ = declare_primitive_type(get_data_type<int64>());
    t_uint64_ = declare_primitive_type(get_data_type<uint64>());
  }
  t_fp32_ = declare_primitive_type(get_data_type<float32>());
  if (device_->get_cap(cap::vk_has_float64)) {
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
  t_v3_uint_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector)
      .add(t_v3_uint_)
      .add_seq(t_uint32_, 3)
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
  return get_const_(dtype, reinterpret_cast<uint64_t *>(&value), cache);
}

Value IRBuilder::uint_immediate_number(const SType &dtype,
                                       uint64_t value,
                                       bool cache) {
  return get_const_(dtype, &value, cache);
}

Value IRBuilder::float_immediate_number(const SType &dtype,
                                        double value,
                                        bool cache) {
  if (data_type_bits(dtype.dt) == 64) {
    return get_const_(dtype, reinterpret_cast<uint64_t *>(&value), cache);
  } else if (data_type_bits(dtype.dt) == 32) {
    float fvalue = static_cast<float>(value);
    uint32_t *ptr = reinterpret_cast<uint32_t *>(&fvalue);
    uint64_t data = ptr[0];
    return get_const_(dtype, &data, cache);
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
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return t_fp32_;
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    if (!device_->get_cap(cap::vk_has_float64))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_fp64_;
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    if (!device_->get_cap(cap::vk_has_int8))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_int8_;
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    if (!device_->get_cap(cap::vk_has_int16))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_int16_;
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return t_int32_;
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    if (!device_->get_cap(cap::vk_has_int64))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_int64_;
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    if (!device_->get_cap(cap::vk_has_int8))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_uint8_;
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    if (!device_->get_cap(cap::vk_has_int16))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_uint16_;
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return t_uint32_;
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    if (!device_->get_cap(cap::vk_has_int64))
      TI_ERROR("Type {} not supported.", dt->to_string());
    return t_uint64_;
  } else {
    TI_ERROR("Type {} not supported.", dt->to_string());
  }
}

SType IRBuilder::get_primitive_buffer_type(const bool struct_compiled,
                                           const DataType &dt) const {
  if (struct_compiled) {
    if (dt->is_primitive(PrimitiveTypeID::f32) &&
        device_->get_cap(cap::vk_has_atomic_float_add)) {
      return t_fp32_;
    } else if (dt->is_primitive(PrimitiveTypeID::f64) &&
               device_->get_cap(cap::vk_has_atomic_float64_add)) {
      return t_fp64_;
    } else if (dt->is_primitive(PrimitiveTypeID::i64) &&
               device_->get_cap(cap::vk_has_atomic_i64)) {
      return t_int64_;
    }
  }
  return t_int32_;
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

SType IRBuilder::get_struct_array_type(const SType &value_type,
                                       uint32_t num_elems) {
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

  if (device_->get_cap(cap::vk_spirv_version) < 0x10300) {
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

Value IRBuilder::buffer_argument(const SType &value_type,
                                 uint32_t descriptor_set,
                                 uint32_t binding) {
  // NOTE: BufferBlock was deprecated in SPIRV 1.3
  // use StorageClassStorageBuffer instead.
  spv::StorageClass storage_class;
  if (device_->get_cap(cap::vk_spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  SType sarr_type = get_struct_array_type(value_type, 0);
  SType ptr_type = get_pointer_type(sarr_type, storage_class);
  Value val = new_value(ptr_type, ValueKind::kStructArrayPtr);
  ib_.begin(spv::OpVariable)
      .add_seq(ptr_type, val, storage_class)
      .commit(&global_);

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
  if (device_->get_cap(cap::vk_spirv_version) < 0x10300) {
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

void IRBuilder::set_work_group_size(const std::array<int, 3> group_size) {
  Value size_x =
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(group_size[0]));
  Value size_y =
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(group_size[1]));
  Value size_z =
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(group_size[2]));

  if (gl_work_group_size.id == 0) {
    gl_work_group_size.id = id_counter_++;
  }
  ib_.begin(spv::OpConstantComposite)
      .add_seq(t_v3_uint_, gl_work_group_size, size_x, size_y, size_z)
      .commit(&global_);
  this->decorate(spv::OpDecorate, gl_work_group_size, spv::DecorationBuiltIn,
                 spv::BuiltInWorkgroupSize);
}

Value IRBuilder::get_num_work_groups(uint32_t dim_index) {
  if (gl_num_work_groups.id == 0) {
    SType ptr_type = this->get_pointer_type(t_v3_uint_, spv::StorageClassInput);
    gl_num_work_groups = new_value(ptr_type, ValueKind::kVectorPtr);
    ib_.begin(spv::OpVariable)
        .add_seq(ptr_type, gl_num_work_groups, spv::StorageClassInput)
        .commit(&global_);
    this->decorate(spv::OpDecorate, gl_num_work_groups, spv::DecorationBuiltIn,
                   spv::BuiltInNumWorkgroups);
  }
  SType pint_type = this->get_pointer_type(t_uint32_, spv::StorageClassInput);
  Value ptr = this->make_value(
      spv::OpAccessChain, pint_type, gl_num_work_groups,
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(dim_index)));

  return this->make_value(spv::OpLoad, t_uint32_, ptr);
}
Value IRBuilder::get_global_invocation_id(uint32_t dim_index) {
  if (gl_global_invocation_id.id == 0) {
    SType ptr_type = this->get_pointer_type(t_v3_uint_, spv::StorageClassInput);
    gl_global_invocation_id = new_value(ptr_type, ValueKind::kVectorPtr);
    ib_.begin(spv::OpVariable)
        .add_seq(ptr_type, gl_global_invocation_id, spv::StorageClassInput)
        .commit(&global_);
    this->decorate(spv::OpDecorate, gl_global_invocation_id,
                   spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
  }
  SType pint_type = this->get_pointer_type(t_uint32_, spv::StorageClassInput);
  Value ptr = this->make_value(
      spv::OpAccessChain, pint_type, gl_global_invocation_id,
      uint_immediate_number(t_uint32_, static_cast<uint64_t>(dim_index)));

  return this->make_value(spv::OpLoad, t_uint32_, ptr);
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
    // a - b * int(float(a) / float(b))
    Value tmp1 = cast(t_fp32_, a);
    Value tmp2 = cast(t_fp32_, b);
    Value tmp3 = make_value(spv::OpFDiv, t_fp32_, tmp1, tmp2);
    // Float division may lose precision
    // FIXME: Could we have a better way to do this?
    Value eps_p = float_immediate_number(t_fp32_, /*+eps=*/1e-5f, false);
    Value eps_n = float_immediate_number(t_fp32_, /*-eps=*/-1e-5f, false);
    Value eps = select(ge(tmp3, eps_p), eps_p, eps_n);
    Value tmp3_float_fixed = make_value(spv::OpFAdd, t_fp32_, tmp3, eps);
    Value tmp4 = cast(a.stype, tmp3_float_fixed);
    Value tmp5 = make_value(spv::OpIMul, a.stype, b, tmp4);
    return make_value(spv::OpISub, a.stype, a, tmp5);
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
  } else if (is_integral(from) && is_signed(from) && is_integral(to) &&
             is_signed(to)) {  // Int -> Int
    return make_value(spv::OpSConvert, dst_type, value);
  } else if (is_integral(from) && is_unsigned(from) && is_integral(to) &&
             is_unsigned(to)) {  // UInt -> UInt
    return make_value(spv::OpUConvert, dst_type, value);
  } else if (is_integral(from) && is_unsigned(from) && is_integral(to) &&
             is_signed(to)) {  // UInt -> Int
    if (data_type_bits(from) != data_type_bits(to)) {
      auto to_signed = [](DataType dt) -> DataType {
        TI_ASSERT(is_unsigned(dt));
        if (dt->is_primitive(PrimitiveTypeID::u8))
          return PrimitiveType::i8;
        else if (dt->is_primitive(PrimitiveTypeID::u16))
          return PrimitiveType::i16;
        else if (dt->is_primitive(PrimitiveTypeID::u32))
          return PrimitiveType::i32;
        else if (dt->is_primitive(PrimitiveTypeID::u64))
          return PrimitiveType::i64;
        else
          return PrimitiveType::unknown;
      };

      value = make_value(spv::OpUConvert, get_primitive_type(to_signed(from)),
                         value);
    }
    return make_value(spv::OpBitcast, dst_type, value);
  } else if (is_integral(from) && is_signed(from) && is_integral(to) &&
             is_unsigned(to)) {  // Int -> UInt
    if (data_type_bits(from) != data_type_bits(to)) {
      value = make_value(spv::OpSConvert, get_primitive_type(to_unsigned(from)),
                         value);
    }
    return make_value(spv::OpBitcast, dst_type, value);
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

Value IRBuilder::load_variable(Value pointer, const SType &res_type) {
  TI_ASSERT(pointer.flag == ValueKind::kVariablePtr ||
            pointer.flag == ValueKind::kStructArrayPtr);
  Value ret = new_value(res_type, ValueKind::kNormal);
  ib_.begin(spv::OpLoad).add_seq(res_type, ret, pointer).commit(&function_);
  return ret;
}
void IRBuilder::store_variable(Value pointer, Value value) {
  TI_ASSERT(pointer.flag == ValueKind::kVariablePtr);
  TI_ASSERT(value.stype.id == pointer.stype.element_type_id);
  ib_.begin(spv::OpStore).add_seq(pointer, value).commit(&function_);
}

void IRBuilder::register_value(std::string name, Value value) {
  auto it = value_name_tbl_.find(name);
  if (it != value_name_tbl_.end()) {
    TI_ERROR("{} is existed.", name);
  }
  this->debug(spv::OpName, value, name);  // Debug info
  value_name_tbl_[name] = value;
}

Value IRBuilder::query_value(std::string name) const {
  auto it = value_name_tbl_.find(name);
  if (it != value_name_tbl_.end()) {
    return it->second;
  }
  TI_ERROR("{} is not existed.", name);
}

Value IRBuilder::float_atomic(AtomicOpType op_type) {
  auto init_atomic_func_ = [&](Value &func, std::string name,
                               std::function<void(Value, Value, Value)>
                                   atomic_op) {
    func.id = id_counter_++;
    func.flag = ValueKind::kFunction;
    debug(spv::OpName, func, name);

    SType addr_ptr_type =
        get_pointer_type(t_int32_, spv::StorageClassStorageBuffer);
    SType data_type = t_fp32_;
    SType float_atomic_func_type;
    float_atomic_func_type.id = id_counter_++;
    float_atomic_func_type.flag = TypeKind::kFunc;
    func.stype = float_atomic_func_type;

    declare_global(spv::OpTypeFunction, float_atomic_func_type, t_fp32_,
                   addr_ptr_type, data_type);

    // function begin
    auto &func_ = atomic_functions_;

    // function header
    ib_.begin(spv::OpFunction)
        .add_seq(t_fp32_, func, 0, float_atomic_func_type)
        .commit(&func_);
    Value addr_ptr = new_value(addr_ptr_type, ValueKind::kStructArrayPtr);
    debug(spv::OpName, addr_ptr, (name + "_addr_ptr").c_str());
    Value data = new_value(data_type, ValueKind::kNormal);
    debug(spv::OpName, data, (name + "_data").c_str());
    ib_.begin(spv::OpFunctionParameter)
        .add_seq(addr_ptr_type, addr_ptr)
        .commit(&func_);
    ib_.begin(spv::OpFunctionParameter).add_seq(data_type, data).commit(&func_);

    auto alloc_var = [&](const SType &type) {
      SType ptr_type = get_pointer_type(type, spv::StorageClassFunction);
      Value ret = new_value(ptr_type, ValueKind::kVariablePtr);
      ib_.begin(spv::OpVariable)
          .add_seq(ptr_type, ret, spv::StorageClassFunction)
          .commit(&func_);

      return ret;
    };

    auto load_var = [&](Value pointer, const SType &res_type) {
      TI_ASSERT(pointer.flag == ValueKind::kVariablePtr ||
                pointer.flag == ValueKind::kStructArrayPtr);
      Value ret = new_value(res_type, ValueKind::kNormal);
      ib_.begin(spv::OpLoad).add_seq(res_type, ret, pointer).commit(&func_);
      return ret;
    };

    auto store_var = [&](Value pointer, Value value) {
      TI_ASSERT(pointer.flag == ValueKind::kVariablePtr);
      TI_ASSERT(value.stype.id == pointer.stype.element_type_id);
      ib_.begin(spv::OpStore).add_seq(pointer, value).commit(&func_);
    };

    // init
    Label init_label = new_label();
    ib_.begin(spv::OpLabel).add(init_label).commit(&func_);
    Value old_val = alloc_var(t_int32_);
    Value new_val = alloc_var(t_int32_);
    Value cas_val = alloc_var(t_int32_);
    Value ok = alloc_var(t_int32_);
    debug(spv::OpName, old_val, (name + "old_val").c_str());
    debug(spv::OpName, new_val, (name + "new_val").c_str());
    debug(spv::OpName, cas_val, (name + "cas_val").c_str());
    debug(spv::OpName, ok, (name + "ok").c_str());

    store_var(old_val, const_i32_zero_);
    store_var(new_val, const_i32_zero_);
    store_var(cas_val, const_i32_zero_);
    store_var(ok, const_i32_zero_);

    // while
    Label head_label = new_label();
    Label body_label = new_label();
    Label continue_label = new_label();
    Label merge_label = new_label();
    Label true_label = new_label();
    ib_.begin(spv::OpBranch).add(head_label).commit(&func_);
    ib_.begin(spv::OpLabel).add(head_label).commit(&func_);
    ib_.begin(spv::OpLoopMerge)
        .add_seq(merge_label, continue_label, spv::LoopControlMaskNone)
        .commit(&func_);
    ib_.begin(spv::OpBranch).add(body_label).commit(&func_);

    // body part
    ib_.begin(spv::OpLabel).add(body_label).commit(&func_);
    Value tmp0 = load_var(ok, t_int32_);
    Value tmp1 = new_value(t_bool_, ValueKind::kNormal);
    ib_.begin(spv::OpIEqual)
        .add_seq(t_bool_, tmp1, tmp0, const_i32_zero_)
        .commit(&func_);
    ib_.begin(spv::OpBranchConditional)
        .add_seq(tmp1, true_label, merge_label)
        .commit(&func_);
    ib_.begin(spv::OpLabel).add(true_label).commit(&func_);
    Value tmp2 = load_var(addr_ptr, t_int32_);
    store_var(old_val, tmp2);
    Value tmp3 = load_var(old_val, t_int32_);
    Value tmp4 = new_value(t_fp32_, ValueKind::kNormal);
    ib_.begin(spv::OpBitcast).add_seq(t_fp32_, tmp4, tmp3).commit(&func_);
    Value tmp5 = new_value(t_fp32_, ValueKind::kNormal);

    // atomic operation
    atomic_op(tmp5, tmp4, data);

    Value tmp6 = new_value(t_int32_, ValueKind::kNormal);
    ib_.begin(spv::OpBitcast).add_seq(t_int32_, tmp6, tmp5).commit(&func_);
    store_var(new_val, tmp6);
    Value tmp7 = load_var(old_val, t_int32_);
    Value tmp8 = load_var(new_val, t_int32_);
    Value tmp9 = new_value(t_int32_, ValueKind::kNormal);
    auto const_u32_1 = uint_immediate_number(t_uint32_, 1);
    auto const_u32_0 = uint_immediate_number(t_uint32_, 0);
    ib_.begin(spv::OpAtomicCompareExchange)
        .add_seq(t_int32_, tmp9, addr_ptr, const_u32_1, const_u32_0,
                 const_u32_0, tmp8, tmp7)
        .commit(&func_);
    store_var(cas_val, tmp9);
    Value tmp10 = load_var(cas_val, t_int32_);
    Value tmp11 = load_var(old_val, t_int32_);
    Value tmp12 = new_value(t_bool_, ValueKind::kNormal);
    ib_.begin(spv::OpIEqual)
        .add_seq(t_bool_, tmp12, tmp10, tmp11)
        .commit(&func_);
    Value tmp13 = new_value(t_int32_, ValueKind::kNormal);
    ib_.begin(spv::OpSelect)
        .add_seq(t_int32_, tmp13, tmp12, const_i32_one_, const_i32_zero_)
        .commit(&func_);
    store_var(ok, tmp13);
    ib_.begin(spv::OpBranch).add(continue_label).commit(&func_);

    // continue part
    ib_.begin(spv::OpLabel).add(continue_label).commit(&func_);
    ib_.begin(spv::OpBranch).add(head_label).commit(&func_);

    // merge part
    ib_.begin(spv::OpLabel).add(merge_label).commit(&func_);
    Value tmp14 = load_var(old_val, t_int32_);
    Value tmp15 = new_value(t_fp32_, ValueKind::kNormal);
    ib_.begin(spv::OpBitcast).add_seq(t_fp32_, tmp15, tmp14).commit(&func_);
    ib_.begin(spv::OpReturnValue).add(tmp15).commit(&func_);
    ib_.begin(spv::OpFunctionEnd).commit(&func_);
    // function end
  };

  if (op_type == AtomicOpType::add) {
    if (float_atomic_add_.id == 0) {
      init_atomic_func_(float_atomic_add_, "float_atomic_add",
                        [&](Value res, Value lhs, Value rhs) {
                          ib_.begin(spv::OpFAdd)
                              .add_seq(t_fp32_, res, lhs, rhs)
                              .commit(&atomic_functions_);
                        });
      any_atomic_ = true;
    }
    return float_atomic_add_;
  } else if (op_type == AtomicOpType::sub) {
    if (float_atomic_sub_.id == 0) {
      init_atomic_func_(float_atomic_sub_, "float_atomic_sub",
                        [&](Value res, Value lhs, Value rhs) {
                          ib_.begin(spv::OpFSub)
                              .add_seq(t_fp32_, res, lhs, rhs)
                              .commit(&atomic_functions_);
                        });
      any_atomic_ = true;
    }
    return float_atomic_sub_;
  } else if (op_type == AtomicOpType::min) {
    if (float_atomic_min_.id == 0) {
      init_atomic_func_(float_atomic_min_, "float_atomic_min",
                        [&](Value res, Value lhs, Value rhs) {
                          Value cond = new_value(t_bool_, ValueKind::kNormal);
                          ib_.begin(spv::OpFOrdLessThan)
                              .add_seq(t_bool_, cond, lhs, rhs)
                              .commit(&atomic_functions_);
                          ib_.begin(spv::OpSelect)
                              .add_seq(t_fp32_, res, cond, lhs, rhs)
                              .commit(&atomic_functions_);
                        });
      any_atomic_ = true;
    }
    return float_atomic_min_;
  } else if (op_type == AtomicOpType::max) {
    if (float_atomic_max_.id == 0) {
      init_atomic_func_(float_atomic_max_, "float_atomic_max",
                        [&](Value res, Value lhs, Value rhs) {
                          Value cond = new_value(t_bool_, ValueKind::kNormal);
                          ib_.begin(spv::OpFOrdGreaterThan)
                              .add_seq(t_bool_, cond, lhs, rhs)
                              .commit(&atomic_functions_);
                          ib_.begin(spv::OpSelect)
                              .add_seq(t_fp32_, res, cond, lhs, rhs)
                              .commit(&atomic_functions_);
                        });
      any_atomic_ = true;
    }
    return float_atomic_max_;
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
  Value tmp0 = load_variable(_rand_x_, t_uint32_);
  Value tmp1 = make_value(spv::OpShiftLeftLogical, t_uint32_, tmp0, _11u);
  Value tmp_t = make_value(spv::OpBitwiseXor, t_uint32_, tmp0, tmp1);  // t
  store_variable(_rand_x_, load_variable(_rand_y_, t_uint32_));
  store_variable(_rand_y_, load_variable(_rand_z_, t_uint32_));
  Value tmp_w = load_variable(_rand_w_, t_uint32_);  // reuse w
  store_variable(_rand_z_, tmp_w);
  Value tmp2 = make_value(spv::OpShiftRightLogical, t_uint32_, tmp_w, _19u);
  Value tmp3 = make_value(spv::OpBitwiseXor, t_uint32_, tmp_w, tmp2);
  Value tmp4 = make_value(spv::OpShiftRightLogical, t_uint32_, tmp_t, _8u);
  Value tmp5 = make_value(spv::OpBitwiseXor, t_uint32_, tmp_t, tmp4);
  Value new_w = make_value(spv::OpBitwiseXor, t_uint32_, tmp3, tmp5);
  store_variable(_rand_w_, new_w);
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

Value IRBuilder::get_const_(const SType &dtype,
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
  _rand_x_ = new_value(local_type, ValueKind::kVariablePtr);
  _rand_y_ = new_value(local_type, ValueKind::kVariablePtr);
  _rand_z_ = new_value(local_type, ValueKind::kVariablePtr);
  _rand_w_ = new_value(local_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpVariable)
      .add_seq(local_type, _rand_x_, spv::StorageClassPrivate)
      .commit(&global_);
  ib_.begin(spv::OpVariable)
      .add_seq(local_type, _rand_y_, spv::StorageClassPrivate)
      .commit(&global_);
  ib_.begin(spv::OpVariable)
      .add_seq(local_type, _rand_z_, spv::StorageClassPrivate)
      .commit(&global_);
  ib_.begin(spv::OpVariable)
      .add_seq(local_type, _rand_w_, spv::StorageClassPrivate)
      .commit(&global_);
  debug(spv::OpName, _rand_x_, "_rand_x");
  debug(spv::OpName, _rand_y_, "_rand_y");
  debug(spv::OpName, _rand_z_, "_rand_z");
  debug(spv::OpName, _rand_w_, "_rand_w");
  SType gtmp_type = get_pointer_type(t_int32_, spv::StorageClassStorageBuffer);
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
  Value _1 = int_immediate_number(t_int32_, 1);
  Value _1024 = int_immediate_number(t_int32_, 1024);

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
      .add_seq(pint_type, tmp0, gl_global_invocation_id,
               uint_immediate_number(t_uint32_, 0))
      .commit(&func_header_);
  Value tmp1 = load_var(tmp0, t_uint32_);
  Value tmp2_ = load_var(rand_gtmp_, t_int32_);
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
  store_var(_rand_x_, tmp8);
  store_var(_rand_y_, _362436069u);
  store_var(_rand_z_, _521288629u);
  store_var(_rand_w_, _88675123u);
  // Yes, this is not an atomic operation, but just fine since no matter
  // how RAND_STATE changes, `gl_GlobalInvocationID.x` can still help
  // us to set different seeds for different threads.
  // Discussion:
  // https://github.com/taichi-dev/taichi/pull/912#discussion_r419021918
  Value tmp9 = load_var(rand_gtmp_, t_int32_);
  Value tmp10 = new_value(t_int32_, ValueKind::kNormal);
  ib_.begin(spv::OpIAdd)
      .add_seq(t_int32_, tmp10, tmp9, _1)
      .commit(&func_header_);
  store_var(rand_gtmp_, tmp10);

  init_rand_ = true;
}

}  // namespace spirv
}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
