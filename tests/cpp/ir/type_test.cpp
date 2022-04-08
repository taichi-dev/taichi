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
  auto ci5 = TypeFactory::get_instance().get_custom_int_type(5, true, i32);
  auto cu11 = TypeFactory::get_instance().get_custom_int_type(11, false, i32);
  auto u16 = TypeFactory::get_instance().get_primitive_int_type(16, false);

  auto bs =
      TypeFactory::get_instance().get_bit_struct_type(u16, {ci5, cu11}, {0, 5});

  EXPECT_EQ(bs->to_string(), "bs(ci5@0, cu11@5)");

  auto ci1 = TypeFactory::get_instance().get_custom_int_type(1, true, i32);
  auto ba = TypeFactory::get_instance().get_bit_array_type(i32, ci1, 32);

  EXPECT_EQ(ba->to_string(), "ba(ci1x32)");
}

}  // namespace lang
}  // namespace taichi
