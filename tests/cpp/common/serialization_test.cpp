#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "taichi/common/core.h"

namespace taichi {
namespace lang {
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

}  // namespace
}  // namespace lang
}  // namespace taichi
