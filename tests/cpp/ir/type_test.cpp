#include "gtest/gtest.h"

#include "taichi/util/testing.h"
#include "taichi/ir/type.h"
#include "taichi/ir/type_factory.h"

namespace taichi {
namespace lang {

TEST(Type, BitTypes) {
  auto f16 =
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f16);
  EXPECT_EQ(f16->to_string(), "f16");
  auto i32 = TypeFactory::get_instance()
                 .get_primitive_type(PrimitiveTypeID::i32)
                 ->as<PrimitiveType>();
  auto qi5 = TypeFactory::get_instance().get_quant_int_type(5, true, i32);
  auto qu11 = TypeFactory::get_instance().get_quant_int_type(11, false, i32);
  auto u16 = TypeFactory::get_instance().get_primitive_int_type(16, false);

  auto bs =
      TypeFactory::get_instance().get_bit_struct_type(u16, {qi5, qu11}, {0, 5}, {false, false}, {-1, -1}, {{}, {}});

  EXPECT_EQ(bs->to_string(), "bs(0: qi5@0, 1: qu11@5)");

  auto qi1 = TypeFactory::get_instance().get_quant_int_type(1, true, i32);
  auto qa = TypeFactory::get_instance().get_quant_array_type(i32, qi1, 32);

  EXPECT_EQ(qa->to_string(), "qa(qi1x32)");
}

}  // namespace lang
}  // namespace taichi
