// Check enums of TiE and taichi::*

#include "taichi/exports/exports.h"
#include "taichi/rhi/arch.h"
#include "taichi/ir/type.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/extension.h"

namespace {

template <typename T>
struct TieCastAttrType {};

template <>
struct TieCastAttrType<int> {
  using Type = int;
};

template <>
struct TieCastAttrType<bool> {
  using Type = bool;
};

template <>
struct TieCastAttrType<std::string> {
  using Type = const char *;
};

template <>
struct TieCastAttrType<taichi::Arch> {
  using Type = int;
};

template <>
struct TieCastAttrType<double> {
  using Type = double;
};

template <>
struct TieCastAttrType<std::size_t> {
  using Type = size_t;
};

template <>
struct TieCastAttrType<taichi::lang::DataType> {
  using Type = TieDataTypeHandle;
};

template <typename T>
using TieCastAttrType_t = typename TieCastAttrType<T>::Type;

}  // namespace

#define TIE_CHECK_TIE_AND_TI_ENUM(TieEnum, tie_enum_name, TaichiEnum, \
                                  ti_enum_name)                       \
  static_assert((std::size_t)tie_enum_name ==                         \
                (std::size_t)TaichiEnum::ti_enum_name);

#define TIE_CHECK_ATTR_TYPE(TaichiStruct, attr_name, attr_type, get_set_type)  \
  static_assert(std::is_same_v<attr_type, decltype(TaichiStruct::attr_name)>); \
  static_assert(std::is_same_v<TieCastAttrType_t<attr_type>, get_set_type>);

#define TIE_PER_ARCH TIE_CHECK_TIE_AND_TI_ENUM
#include "taichi/exports/inc/arch.inc.h"
#undef TIE_PER_ARCH

#define TIE_PER_EXTENSION TIE_CHECK_TIE_AND_TI_ENUM
#include "taichi/exports/inc/extension.inc.h"
#undef TIE_PER_EXTENSION

#define TIE_PER_COMPILE_CONFIG_ATTR TIE_CHECK_ATTR_TYPE
#include "taichi/exports/inc/compile_config.inc.h"
#undef TIE_PER_COMPILE_CONFIG_ATTR
