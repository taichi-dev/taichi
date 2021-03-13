// Definitions of utility functions and enums

#include "taichi/lang_util.h"

#include "taichi/math/linalg.h"
#include "taichi/program/arch.h"
#include "taichi/program/program.h"
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

std::string TypedConstant::stringify() const {
  // TODO: remove the line below after type system upgrade.
  auto dt = this->dt.ptr_removed();
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return fmt::format("{}", val_f32);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return fmt::format("{}", val_i32);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return fmt::format("{}", val_i64);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return fmt::format("{}", val_f64);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return fmt::format("{}", val_i8);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return fmt::format("{}", val_i16);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return fmt::format("{}", val_u8);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return fmt::format("{}", val_u16);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return fmt::format("{}", val_u32);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
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
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return val_f32 == o.val_f32;
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return val_i32 == o.val_i32;
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return val_i64 == o.val_i64;
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return val_f64 == o.val_f64;
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return val_i8 == o.val_i8;
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return val_i16 == o.val_i16;
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return val_u8 == o.val_u8;
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return val_u16 == o.val_u16;
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return val_u32 == o.val_u32;
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
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
  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return val_i32;
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return val_i64;
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return val_i8;
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return val_i16;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

uint64 TypedConstant::val_uint() const {
  TI_ASSERT(is_unsigned(dt));
  if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return val_u32;
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return val_u64;
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return val_u8;
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return val_u16;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

float64 TypedConstant::val_float() const {
  TI_ASSERT(is_real(dt));
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return val_f32;
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
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
