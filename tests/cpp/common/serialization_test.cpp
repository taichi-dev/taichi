#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "taichi/common/core.h"
#include "taichi/ir/type_factory.h"

namespace taichi::lang {
namespace {

class BinIoPair {
 public:
  template <typename T>
  T run(const T &val) {
    BinaryOutputSerializer os;
    os.initialize();
    os(val);  // serialize
    os.finalize();

    T res{};
    BinaryInputSerializer is;
    is.initialize(os.data.data());
    is(res);  // deserialize

    // This is just to make sure TextSerializer also works
    TextSerializer ts;
    ts("val", val);
    return res;
  }
};

enum class EC { Foo, Bar, Baz };

struct Parent {
  struct Child {
    int a{0};
    float b{0.0f};
    bool c{false};

    bool operator==(const Child &other) const {
      return a == other.a && b == other.b && c == other.c;
    }

    TI_IO_DEF(a, b, c);
  };

  std::optional<Child> b;
  std::string c;

  bool operator==(const Parent &other) const {
    return b == other.b && c == other.c;
  }

  TI_IO_DEF(b, c);
};

TEST(Serialization, SplitStr) {
  using namespace detail;

#define STR(...) #__VA_ARGS__
  constexpr auto kDelimN = count_delim(STR(a, bc, def, gh), ',');
  constexpr auto kArr =
      StrDelimSplitter<kDelimN>::make(STR(a, bc, def, gh), ',');
  const std::vector<std::string> expected = {"a", "bc", "def", "gh"};
  EXPECT_EQ(kArr.size(), expected.size());
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(kArr[i], expected[i]);
  }
#undef STR
}

TEST(Serialization, Basic) {
  BinIoPair bp;

  EXPECT_TRUE(bp.run(true));
  EXPECT_FALSE(bp.run(false));

  EXPECT_EQ(bp.run(42), 42);
  EXPECT_FLOAT_EQ(bp.run(42.5f), 42.5f);

  std::string st{"abcde"};
  EXPECT_EQ(bp.run(st), st);

  EXPECT_EQ(bp.run(EC::Foo), EC::Foo);
  EXPECT_EQ(bp.run(EC::Bar), EC::Bar);
  EXPECT_EQ(bp.run(EC::Baz), EC::Baz);

  std::vector<int> vec = {1, 2, 3, 4, 5};
  EXPECT_EQ(bp.run(vec), vec);

  std::optional<std::vector<int>> opt_vec{std::nullopt};
  EXPECT_EQ(bp.run(opt_vec), std::nullopt);
  opt_vec = vec;
  EXPECT_EQ(bp.run(opt_vec).value(), vec);

  std::unordered_map<std::string, int> um = {
      {"a", 1},
      {"b", 2},
      {"c", 3},
  };
  EXPECT_EQ(bp.run(um), um);

  std::map<std::string, int> m = {
      {"a", 1},
      {"b", 2},
      {"c", 3},
  };
  EXPECT_EQ(bp.run(m), m);

  Parent par;
  par.b = Parent::Child{};
  par.b->a = 42;
  par.b->b = 1.23f;
  par.b->c = true;
  par.c = "hello";
  EXPECT_EQ(bp.run(par), par);

  // TODO: Have a proper way to test this...
  TextSerializer ts;
  ts("par", par);
  ts.print();
}

struct MoveOnlyObj {
  int foo{0};
  std::string bar;
  std::unique_ptr<int> ptr{nullptr};

  TI_IO_DEF(foo, bar);
};

TEST(Serialization, MoveOnly) {
  std::unordered_map<std::string, MoveOnlyObj> m;
  m["1"] = MoveOnlyObj{42, "abc", nullptr};
  m["2"] = MoveOnlyObj{100, "def", nullptr};

  BinIoPair bp;
  const auto actual = bp.run(m);
  EXPECT_EQ(actual.size(), m.size());
  const auto &exp_item1 = m.at("1");
  const auto &act_item1 = actual.at("1");
  EXPECT_EQ(act_item1.foo, exp_item1.foo);
  EXPECT_EQ(act_item1.bar, exp_item1.bar);
  EXPECT_EQ(act_item1.ptr, nullptr);
}

TEST(SERIALIZATION, Type) {
  BinIoPair bp;

  // Tests for null type
  Type *null_type = nullptr;
  EXPECT_EQ(bp.run(null_type), null_type);
  auto json_value_null = liong::json::serialize(null_type);
  Type *deserialized_null;
  liong::json::deserialize(json_value_null, deserialized_null);
  EXPECT_EQ(deserialized_null, null_type);

  // Tests for PrimitiveType, TensorType, StructType, PointerType
  auto *int32_type =
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32);
  auto *float32_type =
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32);
  auto *tensor_type =
      TypeFactory::get_instance().get_tensor_type({2, 2}, int32_type);
  auto *struct_type = TypeFactory::get_instance().get_struct_type(
      {{int32_type, "i32", 0}, {tensor_type, "tensor", 4}}, "test");
  auto *pointer_type =
      TypeFactory::get_instance().get_pointer_type((Type *)struct_type, false);
  // The pointer should be the same as the original one
  EXPECT_EQ(bp.run(pointer_type->as<PointerType>()), pointer_type);
  auto json_value = liong::json::serialize(pointer_type->as<PointerType>());
  Type *deserialized;
  liong::json::deserialize(json_value, deserialized);
  EXPECT_EQ(deserialized, pointer_type);

  // Tests for QuantIntType, QuantFloatType, QuantFixedType, BitStructType
  auto *quant_int6_type =
      TypeFactory::get_instance().get_quant_int_type(6, false, int32_type);
  auto *quant_int5_type =
      TypeFactory::get_instance().get_quant_int_type(5, false, int32_type);
  auto *quant_int7_type =
      TypeFactory::get_instance().get_quant_int_type(7, false, int32_type);
  auto *quant_float11_type = TypeFactory::get_instance().get_quant_float_type(
      quant_int5_type, quant_int6_type, float32_type);
  auto *quant_fixed_type = TypeFactory::get_instance().get_quant_fixed_type(
      quant_int7_type, float32_type, 2.0);
  auto *bit_struct_type = TypeFactory::get_instance().get_bit_struct_type(
      int32_type->as<PrimitiveType>(),
      {quant_int6_type, quant_float11_type, quant_fixed_type}, {0, 6, 11},
      {-1, -1, -1}, {{}, {}, {}});
  // BitStructType doesn't have a deduplication mechanism, so we can't compare
  // the pointers
  EXPECT_EQ(bp.run(bit_struct_type)->to_string(), bit_struct_type->to_string());
  auto json_value2 = liong::json::serialize(bit_struct_type);
  const Type *deserialized2;
  liong::json::deserialize(json_value2, deserialized2);
  EXPECT_EQ(deserialized2->to_string(), bit_struct_type->to_string());

  // Tests for QuantArrayType
  auto *quant_array_type = TypeFactory::get_instance().get_quant_array_type(
      int32_type->as<PrimitiveType>(), quant_int6_type, 3);
  // QuantArrayType doesn't have a deduplication mechanism, so we can't compare
  // the pointers
  EXPECT_EQ(bp.run(quant_array_type)->to_string(),
            quant_array_type->to_string());
  auto json_value3 = liong::json::serialize(quant_array_type);
  const Type *deserialized3;
  liong::json::deserialize(json_value3, deserialized3);
  EXPECT_EQ(deserialized3->to_string(), quant_array_type->to_string());
}

struct Foo {
  std::string k;
  int v{-1};

  bool operator==(const Foo &other) const {
    return k == other.k && v == other.v;
  }

  TI_IO_DEF(k, v);
};

TEST(Serialization, JsonSerde) {
  using namespace ::liong::json;

  const auto kCorrectJson = R"({"k":"hello","v":42})";
  const auto kWrongFieldNameJson = R"({"k":"hello","value":42})";
  const auto kWrongFieldTypeJson = R"({"k":"hello","v":"42"})";
  const auto kMissingFieldJson = R"({"k":"hello"})";
  const auto kExtraFieldJson = R"({"k":"hello","v":42,"extra":1})";

  Foo foo, t;
  foo.k = "hello";
  foo.v = 42;

  // Serialize
  EXPECT_EQ(kCorrectJson, print(serialize(foo)));

  // Deserialize (correct)
  deserialize(parse(kCorrectJson), t, true);
  EXPECT_EQ(foo, t);

  // Deserialize (wrong, on strict mode)
  EXPECT_THROW(deserialize(parse(kWrongFieldNameJson), t, true), JsonException);
  EXPECT_THROW(deserialize(parse(kWrongFieldTypeJson), t, true), JsonException);
  EXPECT_THROW(deserialize(parse(kMissingFieldJson), t, true), JsonException);
  EXPECT_THROW(deserialize(parse(kExtraFieldJson), t, true), JsonException);

  // Deserialize (wrong, but on non-strict mode)
  t = Foo{};
  deserialize(parse(kWrongFieldNameJson), t, false);  // no exception
  EXPECT_EQ(foo.k, t.k);
  EXPECT_EQ(-1, t.v);  // default value

  t = Foo{};
  deserialize(parse(kMissingFieldJson), t, false);  // no exception
  EXPECT_EQ(foo.k, t.k);
  EXPECT_EQ(-1, t.v);  // default value

  t = Foo{};
  deserialize(parse(kExtraFieldJson), t, false);  // no exception
  EXPECT_EQ(foo, t);
}

}  // namespace
}  // namespace taichi::lang
