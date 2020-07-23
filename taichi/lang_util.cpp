// Definitions of utility functions and enums

#include "lang_util.h"

#include "taichi/math/linalg.h"
#include "taichi/program/arch.h"
#include "taichi/program/compile_config.h"
#include "taichi/system/timer.h"

TI_NAMESPACE_BEGIN

namespace lang {

CompileConfig default_compile_config;

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
  static std::map<DataType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_DATA_TYPE(i, j) type_names[DataType::i] = #j;
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
  }
  return type_names[t];
}

std::string data_type_format(DataType dt) {
  if (dt == DataType::i32) {
    return "%d";
  } else if (dt == DataType::i64) {
#if defined(TI_PLATFORM_UNIX)
    return "%lld";
#else
    return "%I64d";
#endif
  } else if (dt == DataType::f32) {
    return "%f";
  } else if (dt == DataType::f64) {
    return "%.12f";
  } else {
    TI_NOT_IMPLEMENTED
  }
}

int data_type_size(DataType t) {
  static std::map<DataType, int> type_sizes;
  if (type_sizes.empty()) {
#define REGISTER_DATA_TYPE(i, j) type_sizes[DataType::i] = sizeof(j);
    type_sizes[DataType::f16] = 2;
    REGISTER_DATA_TYPE(f32, float32);
    REGISTER_DATA_TYPE(f64, float64);
    REGISTER_DATA_TYPE(i8, bool);
    REGISTER_DATA_TYPE(i8, int8);
    REGISTER_DATA_TYPE(i16, int16);
    REGISTER_DATA_TYPE(i32, int32);
    REGISTER_DATA_TYPE(i64, int64);
    REGISTER_DATA_TYPE(u8, uint8);
    REGISTER_DATA_TYPE(u16, uint16);
    REGISTER_DATA_TYPE(u32, uint32);
    REGISTER_DATA_TYPE(u64, uint64);
    type_sizes[DataType::gen] = 0;
    type_sizes[DataType::unknown] = -1;
#undef REGISTER_DATA_TYPE
  }
  return type_sizes[t];
}

std::string data_type_short_name(DataType t) {
  static std::map<DataType, std::string> type_names;
  if (type_names.empty()) {
#define PER_TYPE(i) type_names[DataType::i] = #i;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  }
  return type_names[t];
}

std::string snode_type_name(SNodeType t) {
  static std::map<SNodeType, std::string> type_names;
  if (type_names.empty()) {
#define PER_SNODE(i) type_names[SNodeType::i] = #i;
#include "inc/snodes.inc.h"
#undef PER_SNODE
  }
  return type_names[t];
}

bool is_gc_able(SNodeType t) {
  return (t == SNodeType::pointer || t == SNodeType::dynamic);
}

std::string unary_op_type_name(UnaryOpType type) {
  static std::map<UnaryOpType, std::string> type_names;
  if (type_names.empty()) {
#define PER_UNARY_OP(i) type_names[UnaryOpType::i] = #i;
#include "taichi/inc/unary_op.inc.h"

#undef PER_UNARY_OP
  }
  return type_names[type];
}

std::string binary_op_type_name(BinaryOpType type) {
  static std::map<BinaryOpType, std::string> type_names;
  if (type_names.empty()) {
#define PER_BINARY_OP(x) type_names[BinaryOpType::x] = #x;
#include "inc/binary_op.inc.h"
#undef PER_BINARY_OP
  }
  return type_names[type];
}

std::string binary_op_type_symbol(BinaryOpType type) {
  static std::map<BinaryOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i, s) type_names[BinaryOpType::i] = #s;
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
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string ternary_type_name(TernaryOpType type) {
  static std::map<TernaryOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[TernaryOpType::i] = #i;
    REGISTER_TYPE(select);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string atomic_op_type_name(AtomicOpType type) {
  static std::map<AtomicOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[AtomicOpType::i] = #i;
    REGISTER_TYPE(add);
    REGISTER_TYPE(max);
    REGISTER_TYPE(min);
    REGISTER_TYPE(bit_and);
    REGISTER_TYPE(bit_or);
    REGISTER_TYPE(bit_xor);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string snode_op_type_name(SNodeOpType type) {
  static std::map<SNodeOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[SNodeOpType::i] = #i;
    REGISTER_TYPE(is_active);
    REGISTER_TYPE(length);
    REGISTER_TYPE(activate);
    REGISTER_TYPE(deactivate);
    REGISTER_TYPE(append);
    REGISTER_TYPE(clear);
    REGISTER_TYPE(undefined);
#undef REGISTER_TYPE
  }
  return type_names[type];
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

DataType promoted_type(DataType a, DataType b) {
  std::map<std::pair<DataType, DataType>, DataType> mapping;
  if (mapping.empty()) {
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
  return mapping[std::make_pair(a, b)];
}

std::string TypedConstant::stringify() const {
  if (dt == DataType::f32) {
    return fmt::format("{}", val_f32);
  } else if (dt == DataType::i32) {
    return fmt::format("{}", val_i32);
  } else if (dt == DataType::i64) {
    return fmt::format("{}", val_i64);
  } else if (dt == DataType::f64) {
    return fmt::format("{}", val_f64);
  } else if (dt == DataType::i8) {
    return fmt::format("{}", val_i8);
  } else if (dt == DataType::i16) {
    return fmt::format("{}", val_i16);
  } else if (dt == DataType::u8) {
    return fmt::format("{}", val_u8);
  } else if (dt == DataType::u16) {
    return fmt::format("{}", val_u16);
  } else if (dt == DataType::u32) {
    return fmt::format("{}", val_u32);
  } else if (dt == DataType::u64) {
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
  if (dt == DataType::f32) {
    return val_f32 == o.val_f32;
  } else if (dt == DataType::i32) {
    return val_i32 == o.val_i32;
  } else if (dt == DataType::i64) {
    return val_i64 == o.val_i64;
  } else if (dt == DataType::f64) {
    return val_f64 == o.val_f64;
  } else if (dt == DataType::i8) {
    return val_i8 == o.val_i8;
  } else if (dt == DataType::i16) {
    return val_i16 == o.val_i16;
  } else if (dt == DataType::u8) {
    return val_u8 == o.val_u8;
  } else if (dt == DataType::u16) {
    return val_u16 == o.val_u16;
  } else if (dt == DataType::u32) {
    return val_u32 == o.val_u32;
  } else if (dt == DataType::u64) {
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
  if (dt == DataType::i32) {
    return val_i32;
  } else if (dt == DataType::i64) {
    return val_i64;
  } else if (dt == DataType::i8) {
    return val_i8;
  } else if (dt == DataType::i16) {
    return val_i16;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

uint64 TypedConstant::val_uint() const {
  TI_ASSERT(is_unsigned(dt));
  if (dt == DataType::u32) {
    return val_u32;
  } else if (dt == DataType::u64) {
    return val_u64;
  } else if (dt == DataType::u8) {
    return val_u8;
  } else if (dt == DataType::u16) {
    return val_u16;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

float64 TypedConstant::val_float() const {
  TI_ASSERT(is_real(dt));
  if (dt == DataType::f32) {
    return val_f32;
  } else if (dt == DataType::f64) {
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
