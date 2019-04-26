#pragma once

#include "tlang.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

struct Matrix {
  int n, m;
  std::vector<Expr> entries;

  Matrix() {
    n = m = 0;
  }

  bool initialized() {
    return n * m >= 1;
  }

  void initialize(int n, int m) {
    this->n = n;
    this->m = m;
    TC_ASSERT(n * m >= 1);
    entries.resize(n * m);
  }

  explicit Matrix(int n, int m = 1) : n(n), m(m) {
    TC_ASSERT(n * m >= 1);
    entries.resize(n * m);
  }

  explicit Matrix(DataType dt, int n, int m = 1) : Matrix(n, m) {
    fill_global(dt);
  }

  // Initialize vector
  template <int d>
  explicit Matrix(const std::array<Expr, d> &input) {
    entries.resize(d);
    for (int i = 0; i < d; i++) {
      entries[i] = input[i];
    }
  }

  // Initialize vector
  explicit Matrix(const std::vector<float32> &input) : Matrix(input.size(), 1) {
    for (int i = 0; i < (int)input.size(); i++) {
      entries[i] = input[i];
    }
  }

  Matrix(const Matrix &o) : Matrix(o.n, o.m) {
    for (int i = 0; i < n * m; i++) {
      entries[i] = o.entries[i];
    }
  }

  void set(const Matrix &o) {
    initialize(o.n, o.m);
    for (int i = 0; i < n * m; i++) {
      entries[i].set(o.entries[i]);
    }
  }

  Matrix map(const std::function<Expr(const Expr &)> &f) const {
    Matrix ret(n, m);
    for (int i = 0; i < (int)entries.size(); i++) {
      ret.entries[i].set(f(entries[i]));
    }
    return ret;
  }

  Expr &operator()(int i, int j);

  const Expr &operator()(int i, int j) const;

  Expr &operator()(int i) {
    TC_ASSERT(0 <= i && i < n * m);
    TC_ASSERT(n == 1 || m == 1);
    return entries[i];
  }

  const Expr &operator()(int i) const {
    TC_ASSERT(0 <= i && i < n * m);
    TC_ASSERT(n == 1 || m == 1);
    return entries[i];
  }

  Matrix &operator=(const Matrix &o) {
    if (!initialized()) {
      n = o.n;
      m = o.m;
      entries = o.entries;
    } else {
      TC_ASSERT(n == o.n && m == o.m);
      for (int i = 0; i < (int)entries.size(); i++) {
        entries[i] = o.entries[i];
      }
    }
    return *this;
  }

  Matrix element_wise_prod(const Matrix &o) {
    TC_ASSERT(n == o.n && m == o.m);
    Matrix ret(n, m);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        ret(i, j) = (*this)(i, j) * o(i, j);
      }
    }
    return ret;
  }

  Matrix operator[](Expr index) {
    Matrix ret(n, m);
    for (int i = 0; i < n * m; i++) {
      ret.entries[i].set(entries[i][index]);
    }
    return ret;
  }

  Matrix operator[](ExpressionGroup index) {
    Matrix ret(n, m);
    for (int i = 0; i < n * m; i++) {
      ret.entries[i].set(entries[i][index]);
    }
    return ret;
  }

  Expr sum() const {
    Expr ret = entries[0];
    for (int i = 1; i < n * m; i++) {
      ret.set(ret + entries[i]);
    }
    return ret;
  }

  Expr norm2() const {
    Expr ret = entries[0] * entries[0];
    for (int i = 1; i < n * m; i++) {
      ret.set(ret + entries[i] * entries[i]);
    }
    return ret;
  }

  Expr norm() const {
    return sqrt(norm2());
  }

  template <typename T>
  void operator+=(const T &o);
  template <typename T>
  void operator-=(const T &o);
  template <typename T>
  void operator*=(const T &o);

  void fill_global(DataType dt);

  template <typename T>
  Matrix cast_elements() const {
    Matrix ret(this->n, this->m);
    for (int i = 0; i < n * m; i++) {
      ret.entries[i].set(cast<T>(entries[i]));
    }
    return ret;
  }

  Matrix col(int j) const {
    TC_ASSERT(0 <= j && j < m);
    Matrix ret(n, 1);
    for (int i = 0; i < n; i++) {
      ret(i, 0) = (*this)(i, j);
    }
    return ret;
  }

  Matrix eval() const {
    Matrix ret(n, m);
    for (int i = 0; i < (int)entries.size(); i++) {
      ret.entries[i] = entries[i].eval();
    }
    return ret;
  }

  static Matrix identity(int dim);
};

inline Matrix operator*(const Expr &A, const Matrix &B) {
  Matrix C(B.n, B.m);
  for (int i = 0; i < B.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = A * B(i, j);
    }
  }
  return C;
}

inline Matrix operator*(const Matrix &B, const Expr &A) {
  Matrix C(B.n, B.m);
  for (int i = 0; i < B.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = A * B(i, j);
    }
  }
  return C;
}

inline Matrix operator+(const Expr &A, const Matrix &B) {
  Matrix C(B.n, B.m);
  for (int i = 0; i < B.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = A + B(i, j);
    }
  }
  return C;
}

inline Matrix operator/(const Expr &A, const Matrix &B) {
  Matrix C(B.n, B.m);
  for (int i = 0; i < B.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = A / B(i, j);
    }
  }
  return C;
}

inline Matrix operator-(const Expr &A, const Matrix &B) {
  Matrix C(B.n, B.m);
  for (int i = 0; i < B.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = A - B(i, j);
    }
  }
  return C;
}

inline Matrix operator+(const Matrix &B, const Expr &A) {
  Matrix C(B.n, B.m);
  for (int i = 0; i < B.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = B(i, j) + A;
    }
  }
  return C;
}

inline Matrix operator-(const Matrix &B, const Expr &A) {
  Matrix C(B.n, B.m);
  for (int i = 0; i < B.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = B(i, j) - A;
    }
  }
  return C;
}

inline Matrix operator*(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.m == B.n);
  Matrix C(A.n, B.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = A(i, 0) * B(0, j);
      for (int k = 1; k < A.m; k++) {
        C(i, j).set(C(i, j) + A(i, k) * B(k, j));
      }
    }
  }
  return C;
}

inline Matrix operator+(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.n == B.n);
  TC_ASSERT(A.m == B.m);
  Matrix C(A.n, A.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.m; j++) {
      C(i, j) = A(i, j) + B(i, j);
    }
  }
  return C;
}

inline Matrix operator-(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.n == B.n);
  TC_ASSERT(A.m == B.m);
  Matrix C(A.n, A.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.m; j++) {
      C(i, j) = A(i, j) - B(i, j);
    }
  }
  return C;
}

using Vector = Matrix;

inline Matrix operator-(const Matrix &A) {
  Matrix C(A.n, A.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.m; j++) {
      C(i, j) = -A(i, j);
    }
  }
  return C;
}

template <typename T>
void Matrix::operator+=(const T &o) {
  for (int i = 0; i < entries.size(); i++)
    entries[i] += o.entries[i];
}

template <typename T>
void Matrix::operator-=(const T &o) {
  for (int i = 0; i < entries.size(); i++)
    entries[i] -= o.entries[i];
}

template <typename T>
void Matrix::operator*=(const T &o) {
  (*this) = (*this) * o;
}

/*
inline Expr operator-(const Expr &a) {
  return n;
}

inline Expr inv(const Expr &a) {
  auto n = Expr::create(NodeType::inv, a);
  n->data_type = a->data_type;
  return n;
}

inline Float32 lerp(Float a, Float x0, Float x1) {
  return (imm(1.0_f) - a) * x0 + a * x1;
}
*/

inline Matrix sqr(const Matrix &M) {
  return M.map([](Expr e) { return e * e; });
}

inline Matrix exp(const Matrix &M) {
  return M.map([](Expr e) { return exp(e); });
}

inline Matrix log(const Matrix &M) {
  return M.map([](Expr e) { return log(e); });
}

inline Matrix abs(const Matrix &M) {
  return M.map([](Expr e) { return abs(e); });
}

inline Matrix inv(const Matrix &M) {
  return M.map([](Expr e) { return inv(e); });
}

inline Matrix outer_product(Vector a, Vector b) {
  TC_ASSERT(a.m == 1);
  TC_ASSERT(b.m == 1);
  Matrix m(a.n, b.n);
  for (int i = 0; i < a.n; i++) {
    for (int j = 0; j < b.n; j++) {
      m(i, j) = a(i) * b(j);
    }
  }
  return m;
}

inline Expr norm2(const Matrix &mat) {
  return mat.norm2();
}

inline Expr norm(const Matrix &mat) {
  return sqrt(norm2(mat));
}

inline Matrix normalized(const Matrix &mat) {
  auto inv_l = 1.0_f / sqrt(norm2(mat));
  return inv_l * mat;
}

inline Expr clamp(Expr input, Expr l, Expr h) {
  return min(max(input, l), h);
}

inline Matrix cross(const Matrix &a, const Matrix &b) {
  auto c = Vector(3);
  c(0) = a(1) * b(2) - a(2) * b(1);
  c(1) = a(2) * b(0) - a(0) * b(2);
  c(2) = a(0) * b(1) - a(1) * b(0);
  return c;
}

inline Matrix floor(const Matrix &o) {
  Matrix ret(o.n, o.m);
  for (int i = 0; i < o.n; i++) {
    for (int j = 0; j < o.m; j++) {
      ret(i, j).set(floor(o(i, j)));
    }
  }
  return ret;
}

inline Matrix Atomic(const Matrix &dest) {
  // NOTE: dest must be passed by value so that the original
  // expr will not be modified into an atomic one.
  Matrix ret(dest.n, dest.m);
  for (int i = 0; i < (int)dest.entries.size(); i++) {
    ret.entries[i].set(dest.entries[i]);
    ret.entries[i].atomic = true;
  }
  return ret;
}

Matrix& Mutable(Matrix &mat);

Matrix transposed(const Matrix &m);

Matrix diag_matrix(const Matrix &m);

TLANG_NAMESPACE_END
