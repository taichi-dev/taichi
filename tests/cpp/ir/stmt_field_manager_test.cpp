#include "gtest/gtest.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/testing.h"

namespace taichi {
namespace lang {

namespace {
class TestStmt : public Stmt {
 private:
  Stmt *input;
  int a;
  float b;

 public:
  TestStmt(Stmt *input, int a, int b) : input(input), a(a), b(b) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(input, a, b);
};

class TestStmtVector : public Stmt {
 private:
  std::vector<Stmt *> vec1;
  std::vector<Stmt *> vec2;

 public:
  TestStmtVector(const std::vector<Stmt *> &vec1,
                 const std::vector<Stmt *> &vec2)
      : vec1(vec1), vec2(vec2) {
    TI_STMT_REG_FIELDS;
  }

  TI_STMT_DEF_FIELDS(vec1, vec2);
};
}  // namespace

TEST(StmtFieldManager, TestStmtFieldManager) {
  auto a = Stmt::make<TestStmt>(nullptr, 1, 2.0f);

  EXPECT_EQ(a->num_operands(), 1);
  EXPECT_EQ(a->field_manager.fields.size(), 2);

  auto b = Stmt::make<TestStmt>(nullptr, 1, 2.0f);

  EXPECT_EQ(a->field_manager.equal(b->field_manager), true);

  auto c = Stmt::make<TestStmt>(nullptr, 2, 2.1f);

  EXPECT_EQ(a->field_manager.equal(c->field_manager), false);
  // To test two statements are equal: 1) same Stmt type 2) same operands 3)
  // same field_manager
}

TEST(StmtFieldManager, TestStmtFieldManagerWithVector) {
  auto one = Stmt::make<ConstStmt>(TypedConstant(1));
  auto a = Stmt::make<TestStmtVector>(std::vector<Stmt *>(),
                                      std::vector<Stmt *>(1, one.get()));

  EXPECT_EQ(a->num_operands(), 1);
  EXPECT_EQ(a->field_manager.fields.size(), 2);

  auto b = Stmt::make<TestStmtVector>(std::vector<Stmt *>(),
                                      std::vector<Stmt *>(1, one.get()));

  EXPECT_EQ(a->field_manager.equal(b->field_manager), true);

  auto c = Stmt::make<TestStmtVector>(std::vector<Stmt *>(1, one.get()),
                                      std::vector<Stmt *>());

  EXPECT_EQ(a->field_manager.equal(c->field_manager), false);
}

}  // namespace lang
}  // namespace taichi
