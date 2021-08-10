#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;

static void test_comparison_sugar() {
  // we already trust comparisons between tensors, we're simply checking that
  // the sugared versions are doing the same thing
  Tensor<int, 3> t(6, 7, 5);

  t.setRandom();
  // make sure we have at least one value == 0
  t(0,0,0) = 0;

  Tensor<bool,0> b;

#define TEST_TENSOR_EQUAL(e1, e2) \
  b = ((e1) == (e2)).all();       \
  VERIFY(b())

#define TEST_OP(op) TEST_TENSOR_EQUAL(t op 0, t op t.constant(0))

  TEST_OP(==);
  TEST_OP(!=);
  TEST_OP(<=);
  TEST_OP(>=);
  TEST_OP(<);
  TEST_OP(>);
#undef TEST_OP
#undef TEST_TENSOR_EQUAL
}


static void test_scalar_sugar_add_mul() {
  Tensor<float, 3> A(6, 7, 5);
  Tensor<float, 3> B(6, 7, 5);
  A.setRandom();
  B.setRandom();

  const float alpha = 0.43f;
  const float beta = 0.21f;
  const float gamma = 0.14f;

  Tensor<float, 3> R = A.constant(gamma) + A * A.constant(alpha) + B * B.constant(beta);
  Tensor<float, 3> S = A * alpha + B * beta + gamma;
  Tensor<float, 3> T = gamma + alpha * A + beta * B;

  for (int i = 0; i < 6*7*5; ++i) {
    VERIFY_IS_APPROX(R(i), S(i));
    VERIFY_IS_APPROX(R(i), T(i));
  }
}

static void test_scalar_sugar_sub_div() {
  Tensor<float, 3> A(6, 7, 5);
  Tensor<float, 3> B(6, 7, 5);
  A.setRandom();
  B.setRandom();

  const float alpha = 0.43f;
  const float beta = 0.21f;
  const float gamma = 0.14f;
  const float delta = 0.32f;

  Tensor<float, 3> R = A.constant(gamma) - A / A.constant(alpha)
      - B.constant(beta) / B - A.constant(delta);
  Tensor<float, 3> S = gamma - A / alpha - beta / B - delta;

  for (int i = 0; i < 6*7*5; ++i) {
    VERIFY_IS_APPROX(R(i), S(i));
  }
}

void test_cxx11_tensor_sugar()
{
  CALL_SUBTEST(test_comparison_sugar());
  CALL_SUBTEST(test_scalar_sugar_add_mul());
  CALL_SUBTEST(test_scalar_sugar_sub_div());
}
