#include "gtest/gtest.h"

#include "taichi/ir/type_factory.h"

namespace taichi {
namespace lang {

TEST(Type, TypeToString) {
  auto f16 = TypeFactory::get_instance().get_primitive_real_type(16);
  EXPECT_EQ(f16->to_string(), "f16");

  auto f32 = TypeFactory::get_instance().get_primitive_real_type(32);
  auto i32 = TypeFactory::get_instance().get_primitive_int_type(32, true);
  auto u32 = TypeFactory::get_instance().get_primitive_int_type(32, false);
  auto u16 = TypeFactory::get_instance().get_primitive_int_type(16, false);
  auto qi5 = TypeFactory::get_instance().get_quant_int_type(5, true, i32);
  auto qu7 = TypeFactory::get_instance().get_quant_int_type(7, false, i32);
  auto qfl = TypeFactory::get_instance().get_quant_float_type(qi5, qu7, f32);

  auto bs1 = TypeFactory::get_instance().get_bit_struct_type(
      u16, {qi5, qu7}, {0, 5}, {false, false}, {-1, -1}, {{}, {}});
  EXPECT_EQ(bs1->to_string(), "bs(0: qi5@0, 1: qu7@5)");

  auto bs2 = TypeFactory::get_instance().get_bit_struct_type(
      u32, {qu7, qfl, qu7, qfl}, {0, 7, 12, 19}, {false, false, false, false},
      {-1, 0, -1, 2}, {{1}, {}, {3}, {}});
  EXPECT_EQ(bs2->to_string(),
            "bs(0: qu7@0, 1: qfl(d=qi5 e=qu7 c=f32)@7 exp=0, 2: qu7@12, 3: "
            "qfl(d=qi5 e=qu7 c=f32)@19 exp=2)");

  auto bs3 = TypeFactory::get_instance().get_bit_struct_type(
      u32, {qu7, qfl, qfl}, {0, 7, 12}, {false, true, true}, {-1, 0, 0},
      {{1, 2}, {}, {}});
  EXPECT_EQ(bs3->to_string(),
            "bs(0: qu7@0, 1: qfl(d=qi5 e=qu7 c=f32)@7 shared_exp=0, 2: "
            "qfl(d=qi5 e=qu7 c=f32)@12 shared_exp=0)");

  auto qi1 = TypeFactory::get_instance().get_quant_int_type(1, true, i32);
  auto qa = TypeFactory::get_instance().get_quant_array_type(i32, qi1, 32);
  EXPECT_EQ(qa->to_string(), "qa(qi1x32)");
}

}  // namespace lang
}  // namespace taichi
