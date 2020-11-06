#include "taichi/util/testing.h"
#include "taichi/ir/type.h"
#include "taichi/ir/type_factory.h"

TLANG_NAMESPACE_BEGIN

TI_TEST("type") {
  SECTION("bit_types") {
    auto ci5 = TypeFactory::get_instance().get_custom_int_type(5, true);
    auto cu11 = TypeFactory::get_instance().get_custom_int_type(11, false);
    auto u16 = TypeFactory::get_instance().get_primitive_int_type(16, false);

    auto bs = TypeFactory::get_instance().get_bit_struct_type(u16, {ci5, cu11},
                                                              {0, 5});

    TI_CHECK(bs->to_string() == "bs(ci5@0, cu11@5)");

    auto i32 = TypeFactory::get_instance().get_primitive_int_type(32, true);
    auto ci1 = TypeFactory::get_instance().get_custom_int_type(1, true);
    auto ba = TypeFactory::get_instance().get_bit_array_type(i32, ci1, 32);

    TI_CHECK(ba->to_string() == "ba(ci1x32)");
  }
}

TLANG_NAMESPACE_END
