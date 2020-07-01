#include "gtest/gtest.h"

#include "taichi/util/line_appender.h"

namespace taichi {
namespace lang {
namespace {

constexpr auto kSep = "";
TEST(LineAppender, Basic) {
  LineAppender la;
  la.append("1");
  la.append("2");
  la.append("3");
  EXPECT_EQ(la.lines(kSep), "123");

  la.clear_lines();
  EXPECT_EQ(la.lines(kSep), "");
}

TEST(LineAppender, Cursors1) {
  LineAppender la;
  la.append("1");
  auto c1 = la.make_cursor();
  la.append("4");
  la.append("5");
  auto c2 = la.make_cursor();
  la.append("7");
  la.append("8");
  la.rewind_to_cursor(c1);
  la.append("2");
  la.append("3");
  la.rewind_to_cursor(c2);
  la.append("6");
  la.rewind_to_end();
  la.append("9");

  EXPECT_EQ(la.lines(kSep), "123456789");
};

TEST(LineAppender, Cursors2) {
  LineAppender la;
  la.append("1");
  auto c1 = la.make_cursor();
  la.append("7");
  la.append("8");
  la.rewind_to_cursor(c1);
  la.append("2");
  la.append("3");
  auto c2 = la.make_cursor();
  la.append("6");
  la.rewind_to_end();
  la.append("9");
  la.rewind_to_cursor(c2);
  la.append("4");
  la.append("5");

  EXPECT_EQ(la.lines(kSep), "123456789");
};

TEST(LineAppender, ScopedCursor) {
  LineAppender la;
  la.append("1");
  auto c1 = la.make_cursor();
  la.append("4");
  la.append("5");
  {
    ScopedCursor s(la, c1);
    la.append("2");
    la.append("3");
  }
  la.append("6");
  la.append("7");

  EXPECT_EQ(la.lines(kSep), "1234567");
};

}  // namespace
}  // namespace lang
}  // namespace taichi
