#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/testing.h"

TLANG_NAMESPACE_BEGIN

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

TI_TEST("stmt_field_manager") {
  SECTION("test_stmt_field_manager") {
    auto a = Stmt::make<TestStmt>(nullptr, 1, 2.0f);

    TI_CHECK(a->num_operands() == 1);
    TI_CHECK(a->field_manager.fields.size() == 2);

    auto b = Stmt::make<TestStmt>(nullptr, 1, 2.0f);

    TI_CHECK(a->field_manager.equal(b->field_manager) == true);

    auto c = Stmt::make<TestStmt>(nullptr, 2, 2.1f);

    TI_CHECK(a->field_manager.equal(c->field_manager) == false);
    // To test two statements are equal: 1) same Stmt type 2) same operands 3)
    // same field_manager
  }

  SECTION("test_stmt_field_manager_with_vector") {
    auto one = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(1));
    auto a = Stmt::make<TestStmtVector>(std::vector<Stmt *>(),
                                        std::vector<Stmt *>(1, one.get()));

    TI_CHECK(a->num_operands() == 1);
    TI_CHECK(a->field_manager.fields.size() == 2);

    auto b = Stmt::make<TestStmtVector>(std::vector<Stmt *>(),
                                        std::vector<Stmt *>(1, one.get()));

    TI_CHECK(a->field_manager.equal(b->field_manager) == true);

    auto c = Stmt::make<TestStmtVector>(std::vector<Stmt *>(1, one.get()),
                                        std::vector<Stmt *>());

    TI_CHECK(a->field_manager.equal(c->field_manager) == false);
  }
}

TLANG_NAMESPACE_END
