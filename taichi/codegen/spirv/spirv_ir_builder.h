#pragma once

#include <array>

#include <spirv/unified1/spirv.hpp>
#include "taichi/util/lang_util.h"
#include "taichi/ir/type.h"
#include "taichi/util/testing.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/rhi/device.h"
#include "taichi/ir/statements.h"

namespace taichi {
namespace lang {
namespace spirv {

template <bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template <typename T, typename... Args>
  static void run(const F &f, T &&value, Args &&...args) {  // NOLINT(*)
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, (I + 1), F>::run(
        f, std::forward<Args>(args)...);
  }
};

template <std::size_t I, typename F>
struct for_each_dispatcher<true, I, F> {
  static void run(const F &f) {
  }  // NOLINT(*)
};

template <typename F, typename... Args>
inline void for_each(const F &f, Args &&...args) {  // NOLINT(*)
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>::run(
      f, std::forward<Args>(args)...);
}

enum class TypeKind {
  kPrimitive,
  kSNodeStruct,
  kSNodeArray,  // array components of a kSNodeStruct
  kStruct,
  kPtr,
  kFunc,
  kImage
};

// Represent the SPIRV Type
struct SType {
  // The Id to represent type
  uint32_t id{0};

  // corresponding Taichi type/Compiled SNode info
  DataType dt;

  SNodeDescriptor snode_desc;  // TODO: dt/snode_desc only need one at a time
  std::vector<uint32_t> snode_child_type_id;

  TypeKind flag{TypeKind::kPrimitive};

  // Content type id if it is a pointer/struct-array class
  // TODO: SNODE need a vector to store their childrens' element type id
  uint32_t element_type_id{0};

  // The storage class, if it is a pointer
  spv::StorageClass storage_class{spv::StorageClassMax};
};

enum class ValueKind {
  kNormal,
  kConstant,
  kVectorPtr,
  kStructArrayPtr,
  kVariablePtr,
  kPhysicalPtr,
  kTexture,
  kFunction,
  kExtInst
};

// Represent the SPIRV Value
struct Value {
  // The Id to represent type
  uint32_t id{0};
  // The data type
  SType stype;
  // Additional flags about the value
  ValueKind flag{ValueKind::kNormal};
};

// Represent the SPIRV Label
struct Label {
  // The Id to represent label
  uint32_t id{0};
};

// A SPIRV instruction,
//     can be used as handle to modify its content later
class Instr {
 public:
  uint32_t word_count() const {
    return word_count_;
  }

  uint32_t &operator[](uint32_t idx) {
    TI_ASSERT(idx < word_count_);
    return (*data_)[begin_ + idx];
  }

 private:
  friend class InstrBuilder;

  std::vector<uint32_t> *data_{nullptr};
  uint32_t begin_{0};
  uint32_t word_count_{0};
};

// Representation of phi value
struct PhiValue : public Value {
  Instr instr;

  void set_incoming(uint32_t index, const Value &value, const Label &parent) {
    TI_ASSERT(this->stype.id == value.stype.id);
    instr[3 + index * 2] = value.id;
    instr[3 + index * 2 + 1] = parent.id;
  }
};

// Helper class to build SPIRV instruction
class InstrBuilder {
 public:
  InstrBuilder &begin(spv::Op op) {
    TI_ASSERT(data_.size() == 0U);
    op_ = op;
    data_.push_back(0);
    return *this;
  }

#define ADD(var, id)                \
  InstrBuilder &add(const var &v) { \
    data_.push_back(id);            \
    return *this;                   \
  }

  ADD(Value, v.id);
  ADD(SType, v.id);
  ADD(Label, v.id);
  ADD(uint32_t, v);
#undef ADD

  InstrBuilder &add(const std::vector<uint32_t> &v) {
    for (const auto &v0 : v) {
      add(v0);
    }
    return *this;
  }

  InstrBuilder &add(const std::string &v) {
    const uint32_t word_size = sizeof(uint32_t);
    const auto nwords =
        (static_cast<uint32_t>(v.length()) + word_size) / word_size;
    size_t begin = data_.size();
    data_.resize(begin + nwords, 0U);
    std::copy(v.begin(), v.end(), reinterpret_cast<char *>(&data_[begin]));
    return *this;
  }

  template <typename... Args>
  InstrBuilder &add_seq(Args &&...args) {
    AddSeqHelper helper;
    helper.builder = this;
    for_each(helper, std::forward<Args>(args)...);
    return *this;
  }

  Instr commit(std::vector<uint32_t> *seg) {
    Instr ret;
    ret.data_ = seg;
    ret.begin_ = seg->size();
    ret.word_count_ = static_cast<uint32_t>(data_.size());
    data_[0] = op_ | (ret.word_count_ << spv::WordCountShift);
    seg->insert(seg->end(), data_.begin(), data_.end());
    data_.clear();
    return ret;
  }

 private:
  // current op code
  spv::Op op_;
  // The internal data to store code
  std::vector<uint32_t> data_;
  // helper class to support variadic arguments
  struct AddSeqHelper {
    // The reference to builder
    InstrBuilder *builder;
    // invoke function
    template <typename T>
    void operator()(size_t, const T &v) const {
      builder->add(v);
    }
  };
};

// Builder to build up a single SPIR-V module
class IRBuilder {
 public:
  IRBuilder(const Device *device) : device_(device) {
  }

  template <typename... Args>
  void debug(spv::Op op, Args &&...args) {
    ib_.begin(op).add_seq(std::forward<Args>(args)...).commit(&debug_);
  }

  template <typename... Args>
  void execution_mode(Value func, Args &&...args) {
    ib_.begin(spv::OpExecutionMode)
        .add_seq(func, std::forward<Args>(args)...)
        .commit(&exec_mode_);
  }

  template <typename... Args>
  void decorate(spv::Op op, Args &&...args) {
    ib_.begin(op).add_seq(std::forward<Args>(args)...).commit(&decorate_);
  }

  template <typename... Args>
  void declare_global(spv::Op op, Args &&...args) {
    ib_.begin(op).add_seq(std::forward<Args>(args)...).commit(&global_);
  }

  template <typename... Args>
  Instr make_inst(spv::Op op, Args &&...args) {
    return ib_.begin(op)
        .add_seq(std::forward<Args>(args)...)
        .commit(&function_);
  }

  // Initialize header
  void init_header();
  // Initialize the predefined contents
  void init_pre_defs();
  // Get the final binary built from the builder, return The finalized binary
  // instruction
  std::vector<uint32_t> finalize();

  Value ext_inst_import(const std::string &name) {
    Value val = new_value(SType(), ValueKind::kExtInst);
    ib_.begin(spv::OpExtInstImport).add_seq(val, name).commit(&header_);
    return val;
  }

  Label new_label() {
    Label label;
    label.id = id_counter_++;
    return label;
  }

  // Start a new block with given label
  void start_label(Label label) {
    make_inst(spv::OpLabel, label);
    curr_label_ = label;
  }

  Label current_label() const {
    return curr_label_;
  }

  // Make a new SSA value
  template <typename... Args>
  Value make_value(spv::Op op, const SType &out_type, Args &&...args) {
    Value val = new_value(out_type, ValueKind::kNormal);
    make_inst(op, out_type, val, std::forward<Args>(args)...);
    if (out_type.flag == TypeKind::kPtr) {
      val.flag = ValueKind::kVariablePtr;
    }
    return val;
  }

  // Make a phi value
  PhiValue make_phi(const SType &out_type, uint32_t num_incoming);

  // Create Constant Primitive Value
  // cache: if a variable is named, it should not be cached, or the name may
  // have conflict.
  Value int_immediate_number(const SType &dtype,
                             int64_t value,
                             bool cache = true);
  Value uint_immediate_number(const SType &dtype,
                              uint64_t value,
                              bool cache = true);
  Value float_immediate_number(const SType &dtype,
                               double value,
                               bool cache = true);

  // Match zero type
  Value get_zero(const SType &stype) {
    TI_ASSERT(stype.flag == TypeKind::kPrimitive);
    if (is_integral(stype.dt)) {
      if (is_signed(stype.dt)) {
        return int_immediate_number(stype, 0);
      } else {
        return uint_immediate_number(stype, 0);
      }
    } else if (is_real(stype.dt)) {
      return float_immediate_number(stype, 0);
    } else {
      TI_NOT_IMPLEMENTED
      return Value();
    }
  }

  // Get null stype
  SType get_null_type();
  // Get the spirv type for a given Taichi data type
  SType get_primitive_type(const DataType &dt) const;
  // Get the size in bytes of a given Taichi data type
  size_t get_primitive_type_size(const DataType &dt) const;
  // Get the spirv uint type with the same size of a given Taichi data type
  SType get_primitive_uint_type(const DataType &dt) const;
  // Get the Taichi uint type with the same size of a given Taichi data type
  DataType get_taichi_uint_type(const DataType &dt) const;
  // Get the pointer type that points to value_type
  SType get_storage_pointer_type(const SType &value_type);
  // Get the pointer type that points to value_type
  SType get_pointer_type(const SType &value_type,
                         spv::StorageClass storage_class);
  // Get an image type
  SType get_sampled_image_type(const SType &primitive_type, int num_dimensions);
  SType get_storage_image_type(BufferFormat format, int num_dimensions);
  // Get a value_type[num_elems] type
  SType get_array_type(const SType &value_type, uint32_t num_elems);
  // Get a struct{ value_type[num_elems] } type
  SType get_struct_array_type(const SType &value_type, uint32_t num_elems);
  // Construct a struct type
  SType create_struct_type(
      std::vector<std::tuple<SType, std::string, size_t>> &components);

  // Declare buffer argument of function
  Value buffer_struct_argument(const SType &struct_type,
                               uint32_t descriptor_set,
                               uint32_t binding,
                               const std::string &name);
  Value uniform_struct_argument(const SType &struct_type,
                                uint32_t descriptor_set,
                                uint32_t binding,
                                const std::string &name);
  Value buffer_argument(const SType &value_type,
                        uint32_t descriptor_set,
                        uint32_t binding,
                        const std::string &name);
  Value struct_array_access(const SType &res_type, Value buffer, Value index);

  Value texture_argument(int num_channels,
                         int num_dimensions,
                         uint32_t descriptor_set,
                         uint32_t binding);

  Value storage_image_argument(int num_channels,
                               int num_dimensions,
                               uint32_t descriptor_set,
                               uint32_t binding,
                               BufferFormat format);

  Value sample_texture(Value texture_var,
                       const std::vector<Value> &args,
                       Value lod);

  Value fetch_texel(Value texture_var,
                    const std::vector<Value> &args,
                    Value lod);

  Value image_load(Value image_var, const std::vector<Value> &args);

  void image_store(Value image_var, const std::vector<Value> &args);

  // Declare a new function
  // NOTE: only support void kernel function, i.e. main
  Value new_function() {
    return new_value(t_void_func_, ValueKind::kFunction);
  }

  std::vector<Value> global_values;

  // Declare the entry point for a kernel function
  void commit_kernel_function(const Value &func,
                              const std::string &name,
                              std::vector<Value> args,
                              std::array<int, 3> local_size) {
    ib_.begin(spv::OpEntryPoint)
        .add_seq(spv::ExecutionModelGLCompute, func, name);
    for (const auto &arg : args) {
      ib_.add(arg);
    }
    if (device_->get_cap(DeviceCapability::spirv_version) >= 0x10400) {
      for (const auto &v : global_values) {
        ib_.add(v);
      }
    }
    if (gl_global_invocation_id_.id != 0) {
      ib_.add(gl_global_invocation_id_);
    }
    if (gl_num_work_groups_.id != 0) {
      ib_.add(gl_num_work_groups_);
    }
    ib_.commit(&entry_);
    ib_.begin(spv::OpExecutionMode)
        .add_seq(func, spv::ExecutionModeLocalSize, local_size[0],
                 local_size[1], local_size[2])
        .commit(&entry_);
  }

  // Start function scope
  void start_function(const Value &func) {
    // add function declaration to the header
    ib_.begin(spv::OpFunction)
        .add_seq(t_void_, func, 0, t_void_func_)
        .commit(&func_header_);

    spirv::Label start_label = this->new_label();
    ib_.begin(spv::OpLabel).add_seq(start_label).commit(&func_header_);
    curr_label_ = start_label;
  }

  // Declare gl compute shader related methods
  void set_work_group_size(const std::array<int, 3> group_size);
  Value get_work_group_size(uint32_t dim_index);
  Value get_num_work_groups(uint32_t dim_index);
  Value get_local_invocation_id(uint32_t dim_index);
  Value get_global_invocation_id(uint32_t dim_index);
  Value get_subgroup_invocation_id();
  Value get_subgroup_size();

  // Expressions
  Value add(Value a, Value b);
  Value sub(Value a, Value b);
  Value mul(Value a, Value b);
  Value div(Value a, Value b);
  Value mod(Value a, Value b);
  Value eq(Value a, Value b);
  Value ne(Value a, Value b);
  Value lt(Value a, Value b);
  Value le(Value a, Value b);
  Value gt(Value a, Value b);
  Value ge(Value a, Value b);
  Value select(Value cond, Value a, Value b);

  // Create a cast that cast value to dst_type
  Value cast(const SType &dst_type, Value value);

  // Create a GLSL450 call
  template <typename... Args>
  Value call_glsl450(const SType &ret_type, uint32_t inst_id, Args &&...args) {
    Value val = new_value(ret_type, ValueKind::kNormal);
    ib_.begin(spv::OpExtInst)
        .add_seq(ret_type, val, ext_glsl450_, inst_id)
        .add_seq(std::forward<Args>(args)...)
        .commit(&function_);
    return val;
  }

  // Local allocate, load, store methods
  Value alloca_variable(const SType &type);
  Value alloca_workgroup_array(const SType &type);
  Value load_variable(Value pointer, const SType &res_type);
  void store_variable(Value pointer, Value value);

  // Register name to corresponding Value/VariablePointer
  void register_value(std::string name, Value value);
  // Query Value/VariablePointer by name
  Value query_value(std::string name) const;
  // Check whether a value has been evaluated
  bool check_value_existence(const std::string &name) const;
  // Create a new SSA value
  Value new_value(const SType &type, ValueKind flag) {
    Value val;
    val.id = id_counter_++;
    val.stype = type;
    val.flag = flag;
    return val;
  }

  // Support easy access to trivial data types
  SType i64_type() const {
    return t_int64_;
  }
  SType u64_type() const {
    return t_uint64_;
  }
  SType f64_type() const {
    return t_fp64_;
  }

  SType i32_type() const {
    return t_int32_;
  }
  SType u32_type() const {
    return t_uint32_;
  }
  SType f32_type() const {
    return t_fp32_;
  }

  SType i16_type() const {
    return t_int16_;
  }
  SType u16_type() const {
    return t_uint16_;
  }
  SType f16_type() const {
    return t_fp16_;
  }

  SType i8_type() const {
    return t_int8_;
  }
  SType u8_type() const {
    return t_uint8_;
  }

  SType bool_type() const {
    return t_bool_;
  }

  // quick cache for const zero/one i32
  Value const_i32_zero_;
  Value const_i32_one_;

  // Use force-inline float atomic helper function
  Value float_atomic(AtomicOpType op_type, Value addr_ptr, Value data);
  Value rand_u32(Value global_tmp_);
  Value rand_f32(Value global_tmp_);
  Value rand_i32(Value global_tmp_);

 private:
  Value get_const(const SType &dtype, const uint64_t *pvalue, bool cache);
  SType declare_primitive_type(DataType dt);

  void init_random_function(Value global_tmp_);

  const Device *device_;

  // internal instruction builder
  InstrBuilder ib_;
  // Current label
  Label curr_label_;
  // The current maximum id
  uint32_t id_counter_{1};

  // glsl 450 extension
  Value ext_glsl450_;

  SType t_bool_;
  SType t_int8_;
  SType t_int16_;
  SType t_int32_;
  SType t_int64_;
  SType t_uint8_;
  SType t_uint16_;
  SType t_uint32_;
  SType t_uint64_;
  SType t_fp16_;
  SType t_fp32_;
  SType t_fp64_;
  SType t_void_;
  SType t_void_func_;
  // gl compute shader related type(s) and variables
  SType t_v2_int_;
  SType t_v3_int_;
  SType t_v3_uint_;
  SType t_v4_fp32_;
  SType t_v3_fp32_;
  SType t_v2_fp32_;
  Value gl_global_invocation_id_;
  Value gl_local_invocation_id_;
  Value gl_num_work_groups_;
  Value gl_work_group_size_;
  Value subgroup_local_invocation_id_;
  Value subgroup_size_;

  // Random function and variables
  bool init_rand_{false};
  Value rand_x_;
  Value rand_y_;
  Value rand_z_;
  Value rand_w_;  // per-thread local variable

  // map from value to its pointer type
  std::map<std::pair<uint32_t, spv::StorageClass>, SType> pointer_type_tbl_;
  std::map<std::pair<uint32_t, int>, SType> sampled_image_ptr_tbl_;
  std::map<std::pair<BufferFormat, int>, SType> storage_image_ptr_tbl_;

  // map from constant int to its value
  std::map<std::pair<uint32_t, uint64_t>, Value> const_tbl_;
  // map from raw_name(string) to Value
  std::unordered_map<std::string, Value> value_name_tbl_;

  // Header segment, include import
  std::vector<uint32_t> header_;
  // engtry point segment
  std::vector<uint32_t> entry_;
  // Header segment
  std::vector<uint32_t> exec_mode_;
  // Debug segment
  std::vector<uint32_t> debug_;
  // Annotation segment
  std::vector<uint32_t> decorate_;
  // Global segment: types, variables, types
  std::vector<uint32_t> global_;
  // Function header segment
  std::vector<uint32_t> func_header_;
  // Main Function segment
  std::vector<uint32_t> function_;
};
}  // namespace spirv
}  // namespace lang
}  // namespace taichi
