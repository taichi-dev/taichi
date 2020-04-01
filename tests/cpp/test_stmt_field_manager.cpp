#include "taichi/ir/ir.h"
#include "taichi/common/testing.h"

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

TI_TEST("test_stmt_field_manager") {
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

TLANG_NAMESPACE_END
