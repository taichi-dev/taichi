

#include "lang_util.h"

// Definitions of utility functions and enums

#include "lang_util.h"

#include "taichi/math/linalg.h"
#include "taichi/program/arch.h"
#include "taichi/program/compile_config.h"
#include "taichi/system/timer.h"

TI_NAMESPACE_BEGIN

namespace lang {

CompileConfig default_compile_config;

#define PER_TYPE(x) static DataTypeNode *x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

real get_cpu_frequency() {
  static real cpu_frequency = 0;
  if (cpu_frequency == 0) {
    uint64 cycles = Time::get_cycles();
    Time::sleep(1);
    uint64 elapsed_cycles = Time::get_cycles() - cycles;
    auto frequency = real(std::round(elapsed_cycles / 1e8_f64) / 10.0_f64);
    TI_INFO("CPU frequency = {:.2f} GHz ({} cycles per second)", frequency,
            elapsed_cycles);
    cpu_frequency = frequency;
  }
  return cpu_frequency;
}

real default_measurement_time = 1;

#define PER_TYPE(x)          \
  DataType DataTypeNode::x = \
      new PrimitiveTypeNode(PrimitiveTypeNode::primitive_type::x);
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

DataType PrimitiveTypeNode::get(primitive_type t) {
  if (false) {
  }
#define PER_TYPE(x) else if (t == primitive_type::x) return DataTypeNode::x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  else {
    TI_NOT_IMPLEMENTED
  }
}

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second) {
  if (time_second == 0) {
    target();
    return std::numeric_limits<real>::quiet_NaN();
  }
  // first make rough estimate of run time.
  int64 batch_size = 1;
  while (true) {
    float64 t = Time::get_time();
    for (int64 i = 0; i < batch_size; i++) {
      target();
    }
    t = Time::get_time() - t;
    if (t < 0.05 * time_second) {
      batch_size *= 2;
    } else {
      break;
    }
  }

  int64 total_batches = 0;
  float64 start_t = Time::get_time();
  while (Time::get_time() - start_t < time_second) {
    for (int i = 0; i < batch_size; i++) {
      target();
    }
    total_batches += batch_size;
  }
  auto elasped_cycles =
      (Time::get_time() - start_t) * 1e9_f64 * get_cpu_frequency();
  return elasped_cycles / float64(total_batches * elements_per_call);
}

std::string data_type_name(DataType t) {
#define REGISTER_DATA_TYPE(i, j) else if (t == DataTypeNode::i) return #j
  if (false) {
  }
  REGISTER_DATA_TYPE(f16, float16);
  REGISTER_DATA_TYPE(f32, float32);
  REGISTER_DATA_TYPE(f64, float64);
  REGISTER_DATA_TYPE(u1, int1);
  REGISTER_DATA_TYPE(i8, int8);
  REGISTER_DATA_TYPE(i16, int16);
  REGISTER_DATA_TYPE(i32, int32);
  REGISTER_DATA_TYPE(i64, int64);
  REGISTER_DATA_TYPE(u8, uint8);
  REGISTER_DATA_TYPE(u16, uint16);
  REGISTER_DATA_TYPE(u32, uint32);
  REGISTER_DATA_TYPE(u64, uint64);
  REGISTER_DATA_TYPE(gen, generic);
  REGISTER_DATA_TYPE(unknown, unknown);

#undef REGISTER_DATA_TYPE
  else TI_NOT_IMPLEMENTED
}

std::string data_type_format(DataType dt) {
  if (dt == DataTypeNode::i32) {
    return "%d";
  } else if (dt == DataTypeNode::i64) {
#if defined(TI_PLATFORM_UNIX)
    return "%lld";
#else
    return "%I64d";
#endif
  } else if (dt == DataTypeNode::f32) {
    return "%f";
  } else if (dt == DataTypeNode::f64) {
    return "%.12f";
  } else {
    TI_NOT_IMPLEMENTED
  }
}

int data_type_size(DataType t) {
  if (false) {
  } else if (t == DataTypeNode::f16)
    return 2;
  else if (t == DataTypeNode::gen)
    return 0;
  else if (t == DataTypeNode::unknown)
    return -1;

#define REGISTER_DATA_TYPE(i, j) else if (t == DataTypeNode::i) return sizeof(j)

  REGISTER_DATA_TYPE(f32, float32);
  REGISTER_DATA_TYPE(f64, float64);
  REGISTER_DATA_TYPE(i8, int8);
  REGISTER_DATA_TYPE(i16, int16);
  REGISTER_DATA_TYPE(i32, int32);
  REGISTER_DATA_TYPE(i64, int64);
  REGISTER_DATA_TYPE(u8, uint8);
  REGISTER_DATA_TYPE(u16, uint16);
  REGISTER_DATA_TYPE(u32, uint32);
  REGISTER_DATA_TYPE(u64, uint64);

#undef REGISTER_DATA_TYPE
  else {
    TI_NOT_IMPLEMENTED
  }
}

std::string data_type_short_name(DataType t) {
  if (false) {
  }
#define PER_TYPE(i) else if (t == DataTypeNode::i) return #i;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  else
    TI_NOT_IMPLEMENTED
}  // namespace lang

std::string snode_type_name(SNodeType t) {
  switch (t) {
#define PER_SNODE(i) \
  case SNodeType::i: \
    return #i;

#include "inc/snodes.inc.h"

#undef PER_SNODE
    default:
      TI_NOT_IMPLEMENTED
  }
}

bool is_gc_able(SNodeType t) {
  return (t == SNodeType::pointer || t == SNodeType::dynamic);
}

std::string unary_op_type_name(UnaryOpType type) {
  switch (type) {
#define PER_UNARY_OP(i) \
  case UnaryOpType::i:  \
    return #i;

#include "taichi/inc/unary_op.inc.h"

#undef PER_UNARY_OP
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string binary_op_type_name(BinaryOpType type) {
  switch (type) {
#define PER_BINARY_OP(x) \
  case BinaryOpType::x:  \
    return #x;

#include "inc/binary_op.inc.h"

#undef PER_BINARY_OP
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string binary_op_type_symbol(BinaryOpType type) {
  switch (type) {
#define REGISTER_TYPE(i, s) \
  case BinaryOpType::i:     \
    return #s;

    REGISTER_TYPE(mul, *);
    REGISTER_TYPE(add, +);
    REGISTER_TYPE(sub, -);
    REGISTER_TYPE(div, /);
    REGISTER_TYPE(truediv, /);
    REGISTER_TYPE(floordiv, /);
    REGISTER_TYPE(mod, %);
    REGISTER_TYPE(max, max);
    REGISTER_TYPE(min, min);
    REGISTER_TYPE(atan2, atan2);
    REGISTER_TYPE(cmp_lt, <);
    REGISTER_TYPE(cmp_le, <=);
    REGISTER_TYPE(cmp_gt, >);
    REGISTER_TYPE(cmp_ge, >=);
    REGISTER_TYPE(cmp_ne, !=);
    REGISTER_TYPE(cmp_eq, ==);
    REGISTER_TYPE(bit_and, &);
    REGISTER_TYPE(bit_or, |);
    REGISTER_TYPE(bit_xor, ^);
    REGISTER_TYPE(pow, pow);
    REGISTER_TYPE(bit_shl, <<);
    REGISTER_TYPE(bit_sar, >>);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string ternary_type_name(TernaryOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case TernaryOpType::i: \
    return #i;

    REGISTER_TYPE(select);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string atomic_op_type_name(AtomicOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case AtomicOpType::i:  \
    return #i;

    REGISTER_TYPE(add);
    REGISTER_TYPE(sub);
    REGISTER_TYPE(max);
    REGISTER_TYPE(min);
    REGISTER_TYPE(bit_and);
    REGISTER_TYPE(bit_or);
    REGISTER_TYPE(bit_xor);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

BinaryOpType atomic_to_binary_op_type(AtomicOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case AtomicOpType::i:  \
    return BinaryOpType::i;

    REGISTER_TYPE(add);
    REGISTER_TYPE(sub);
    REGISTER_TYPE(max);
    REGISTER_TYPE(min);
    REGISTER_TYPE(bit_and);
    REGISTER_TYPE(bit_or);
    REGISTER_TYPE(bit_xor);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string snode_op_type_name(SNodeOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case SNodeOpType::i:   \
    return #i;

    REGISTER_TYPE(is_active);
    REGISTER_TYPE(length);
    REGISTER_TYPE(activate);
    REGISTER_TYPE(deactivate);
    REGISTER_TYPE(append);
    REGISTER_TYPE(clear);
    REGISTER_TYPE(undefined);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

bool command_exist(const std::string &command) {
#if defined(TI_PLATFORM_UNIX)
  if (std::system(fmt::format("which {} > /dev/null 2>&1", command).c_str())) {
    return false;
  } else {
    return true;
  }
#else
  if (std::system(fmt::format("where {} >nul 2>nul", command).c_str())) {
    return false;
  } else {
    return true;
  }
#endif
}

namespace {
class TypePromotionMapping {
 public:
  TypePromotionMapping() {
#define TRY_SECOND(x, y)                                            \
  mapping[std::make_pair(get_data_type<x>(), get_data_type<y>())] = \
      get_data_type<decltype(std::declval<x>() + std::declval<y>())>();
#define TRY_FIRST(x)      \
  TRY_SECOND(x, float32); \
  TRY_SECOND(x, float64); \
  TRY_SECOND(x, int8);    \
  TRY_SECOND(x, int16);   \
  TRY_SECOND(x, int32);   \
  TRY_SECOND(x, int64);   \
  TRY_SECOND(x, uint8);   \
  TRY_SECOND(x, uint16);  \
  TRY_SECOND(x, uint32);  \
  TRY_SECOND(x, uint64);

    TRY_FIRST(float32);
    TRY_FIRST(float64);
    TRY_FIRST(int8);
    TRY_FIRST(int16);
    TRY_FIRST(int32);
    TRY_FIRST(int64);
    TRY_FIRST(uint8);
    TRY_FIRST(uint16);
    TRY_FIRST(uint32);
    TRY_FIRST(uint64);
  }
  DataType query(DataType x, DataType y) {
    return mapping[std::make_pair(x, y)];
  }

 private:
  std::map<std::pair<DataType, DataType>, DataType> mapping;
};
TypePromotionMapping type_promotion_mapping;
}  // namespace

DataType promoted_type(DataType a, DataType b) {
  return type_promotion_mapping.query(a, b);
}

std::string TypedConstant::stringify() const {
  if (dt == DataTypeNode::f32) {
    return fmt::format("{}", val_f32);
  } else if (dt == DataTypeNode::i32) {
    return fmt::format("{}", val_i32);
  } else if (dt == DataTypeNode::i64) {
    return fmt::format("{}", val_i64);
  } else if (dt == DataTypeNode::f64) {
    return fmt::format("{}", val_f64);
  } else if (dt == DataTypeNode::i8) {
    return fmt::format("{}", val_i8);
  } else if (dt == DataTypeNode::i16) {
    return fmt::format("{}", val_i16);
  } else if (dt == DataTypeNode::u8) {
    return fmt::format("{}", val_u8);
  } else if (dt == DataTypeNode::u16) {
    return fmt::format("{}", val_u16);
  } else if (dt == DataTypeNode::u32) {
    return fmt::format("{}", val_u32);
  } else if (dt == DataTypeNode::u64) {
    return fmt::format("{}", val_u64);
  } else {
    TI_P(data_type_name(dt));
    TI_NOT_IMPLEMENTED
    return "";
  }
}

bool TypedConstant::equal_type_and_value(const TypedConstant &o) const {
  if (dt != o.dt)
    return false;
  if (dt == DataTypeNode::f32) {
    return val_f32 == o.val_f32;
  } else if (dt == DataTypeNode::i32) {
    return val_i32 == o.val_i32;
  } else if (dt == DataTypeNode::i64) {
    return val_i64 == o.val_i64;
  } else if (dt == DataTypeNode::f64) {
    return val_f64 == o.val_f64;
  } else if (dt == DataTypeNode::i8) {
    return val_i8 == o.val_i8;
  } else if (dt == DataTypeNode::i16) {
    return val_i16 == o.val_i16;
  } else if (dt == DataTypeNode::u8) {
    return val_u8 == o.val_u8;
  } else if (dt == DataTypeNode::u16) {
    return val_u16 == o.val_u16;
  } else if (dt == DataTypeNode::u32) {
    return val_u32 == o.val_u32;
  } else if (dt == DataTypeNode::u64) {
    return val_u64 == o.val_u64;
  } else {
    TI_NOT_IMPLEMENTED
    return false;
  }
}

int32 &TypedConstant::val_int32() {
  TI_ASSERT(get_data_type<int32>() == dt);
  return val_i32;
}

float32 &TypedConstant::val_float32() {
  TI_ASSERT(get_data_type<float32>() == dt);
  return val_f32;
}

int64 &TypedConstant::val_int64() {
  TI_ASSERT(get_data_type<int64>() == dt);
  return val_i64;
}

float64 &TypedConstant::val_float64() {
  TI_ASSERT(get_data_type<float64>() == dt);
  return val_f64;
}

int8 &TypedConstant::val_int8() {
  TI_ASSERT(get_data_type<int8>() == dt);
  return val_i8;
}

int16 &TypedConstant::val_int16() {
  TI_ASSERT(get_data_type<int16>() == dt);
  return val_i16;
}

uint8 &TypedConstant::val_uint8() {
  TI_ASSERT(get_data_type<uint8>() == dt);
  return val_u8;
}

uint16 &TypedConstant::val_uint16() {
  TI_ASSERT(get_data_type<uint16>() == dt);
  return val_u16;
}

uint32 &TypedConstant::val_uint32() {
  TI_ASSERT(get_data_type<uint32>() == dt);
  return val_u32;
}

uint64 &TypedConstant::val_uint64() {
  TI_ASSERT(get_data_type<uint64>() == dt);
  return val_u64;
}

int64 TypedConstant::val_int() const {
  TI_ASSERT(is_signed(dt));
  if (dt == DataTypeNode::i32) {
    return val_i32;
  } else if (dt == DataTypeNode::i64) {
    return val_i64;
  } else if (dt == DataTypeNode::i8) {
    return val_i8;
  } else if (dt == DataTypeNode::i16) {
    return val_i16;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

uint64 TypedConstant::val_uint() const {
  TI_ASSERT(is_unsigned(dt));
  if (dt == DataTypeNode::u32) {
    return val_u32;
  } else if (dt == DataTypeNode::u64) {
    return val_u64;
  } else if (dt == DataTypeNode::u8) {
    return val_u8;
  } else if (dt == DataTypeNode::u16) {
    return val_u16;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

float64 TypedConstant::val_float() const {
  TI_ASSERT(is_real(dt));
  if (dt == DataTypeNode::f32) {
    return val_f32;
  } else if (dt == DataTypeNode::f64) {
    return val_f64;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

float64 TypedConstant::val_cast_to_float64() const {
  if (is_real(dt))
    return val_float();
  else if (is_signed(dt))
    return val_int();
  else if (is_unsigned(dt))
    return val_uint();
  else {
    TI_NOT_IMPLEMENTED
  }
}

}  // namespace lang

void initialize_benchmark() {
  // CoreState::set_trigger_gdb_when_crash(true);
  lang::get_cpu_frequency();
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;
#if defined(TI_PLATFORM_LINUX)
  std::ifstream noturbo("/sys/devices/system/cpu/intel_pstate/no_turbo");
  char c;
  noturbo >> c;
  TI_WARN_IF(c != '1',
             "You seem to be running the benchmark with Intel Turboboost.");
#endif
}

TI_NAMESPACE_END
