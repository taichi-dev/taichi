// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/AutoDiff>

template<typename Scalar>
EIGEN_DONT_INLINE Scalar foo(const Scalar& x, const Scalar& y)
{
  using namespace std;
//   return x+std::sin(y);
  EIGEN_ASM_COMMENT("mybegin");
  // pow(float, int) promotes to pow(double, double)
  return x*2 - 1 + static_cast<Scalar>(pow(1+x,2)) + 2*sqrt(y*y+0) - 4 * sin(0+x) + 2 * cos(y+0) - exp(Scalar(-0.5)*x*x+0);
  //return x+2*y*x;//x*2 -std::pow(x,2);//(2*y/x);// - y*2;
  EIGEN_ASM_COMMENT("myend");
}

template<typename Vector>
EIGEN_DONT_INLINE typename Vector::Scalar foo(const Vector& p)
{
  typedef typename Vector::Scalar Scalar;
  return (p-Vector(Scalar(-1),Scalar(1.))).norm() + (p.array() * p.array()).sum() + p.dot(p);
}

template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct TestFunc1
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  int m_inputs, m_values;

  TestFunc1() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  TestFunc1(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  template<typename T>
  void operator() (const Matrix<T,InputsAtCompileTime,1>& x, Matrix<T,ValuesAtCompileTime,1>* _v) const
  {
    Matrix<T,ValuesAtCompileTime,1>& v = *_v;

    v[0] = 2 * x[0] * x[0] + x[0] * x[1];
    v[1] = 3 * x[1] * x[0] + 0.5 * x[1] * x[1];
    if(inputs()>2)
    {
      v[0] += 0.5 * x[2];
      v[1] += x[2];
    }
    if(values()>2)
    {
      v[2] = 3 * x[1] * x[0] * x[0];
    }
    if (inputs()>2 && values()>2)
      v[2] *= x[2];
  }

  void operator() (const InputType& x, ValueType* v, JacobianType* _j) const
  {
    (*this)(x, v);

    if(_j)
    {
      JacobianType& j = *_j;

      j(0,0) = 4 * x[0] + x[1];
      j(1,0) = 3 * x[1];

      j(0,1) = x[0];
      j(1,1) = 3 * x[0] + 2 * 0.5 * x[1];

      if (inputs()>2)
      {
        j(0,2) = 0.5;
        j(1,2) = 1;
      }
      if(values()>2)
      {
        j(2,0) = 3 * x[1] * 2 * x[0];
        j(2,1) = 3 * x[0] * x[0];
      }
      if (inputs()>2 && values()>2)
      {
        j(2,0) *= x[2];
        j(2,1) *= x[2];

        j(2,2) = 3 * x[1] * x[0] * x[0];
        j(2,2) = 3 * x[1] * x[0] * x[0];
      }
    }
  }
};


#if EIGEN_HAS_VARIADIC_TEMPLATES
/* Test functor for the C++11 features. */
template <typename Scalar>
struct integratorFunctor
{
    typedef Matrix<Scalar, 2, 1> InputType;
    typedef Matrix<Scalar, 2, 1> ValueType;

    /*
     * Implementation starts here.
     */
    integratorFunctor(const Scalar gain) : _gain(gain) {}
    integratorFunctor(const integratorFunctor& f) : _gain(f._gain) {}
    const Scalar _gain;

    template <typename T1, typename T2>
    void operator() (const T1 &input, T2 *output, const Scalar dt) const
    {
        T2 &o = *output;

        /* Integrator to test the AD. */
        o[0] = input[0] + input[1] * dt * _gain;
        o[1] = input[1] * _gain;
    }

    /* Only needed for the test */
    template <typename T1, typename T2, typename T3>
    void operator() (const T1 &input, T2 *output, T3 *jacobian, const Scalar dt) const
    {
        T2 &o = *output;

        /* Integrator to test the AD. */
        o[0] = input[0] + input[1] * dt * _gain;
        o[1] = input[1] * _gain;

        if (jacobian)
        {
            T3 &j = *jacobian;

            j(0, 0) = 1;
            j(0, 1) = dt * _gain;
            j(1, 0) = 0;
            j(1, 1) = _gain;
        }
    }

};

template<typename Func> void forward_jacobian_cpp11(const Func& f)
{
    typedef typename Func::ValueType::Scalar Scalar;
    typedef typename Func::ValueType ValueType;
    typedef typename Func::InputType InputType;
    typedef typename AutoDiffJacobian<Func>::JacobianType JacobianType;

    InputType x = InputType::Random(InputType::RowsAtCompileTime);
    ValueType y, yref;
    JacobianType j, jref;

    const Scalar dt = internal::random<double>();

    jref.setZero();
    yref.setZero();
    f(x, &yref, &jref, dt);

    //std::cerr << "y, yref, jref: " << "\n";
    //std::cerr << y.transpose() << "\n\n";
    //std::cerr << yref << "\n\n";
    //std::cerr << jref << "\n\n";

    AutoDiffJacobian<Func> autoj(f);
    autoj(x, &y, &j, dt);

    //std::cerr << "y j (via autodiff): " << "\n";
    //std::cerr << y.transpose() << "\n\n";
    //std::cerr << j << "\n\n";

    VERIFY_IS_APPROX(y, yref);
    VERIFY_IS_APPROX(j, jref);
}
#endif

template<typename Func> void forward_jacobian(const Func& f)
{
    typename Func::InputType x = Func::InputType::Random(f.inputs());
    typename Func::ValueType y(f.values()), yref(f.values());
    typename Func::JacobianType j(f.values(),f.inputs()), jref(f.values(),f.inputs());

    jref.setZero();
    yref.setZero();
    f(x,&yref,&jref);
//     std::cerr << y.transpose() << "\n\n";;
//     std::cerr << j << "\n\n";;

    j.setZero();
    y.setZero();
    AutoDiffJacobian<Func> autoj(f);
    autoj(x, &y, &j);
//     std::cerr << y.transpose() << "\n\n";;
//     std::cerr << j << "\n\n";;

    VERIFY_IS_APPROX(y, yref);
    VERIFY_IS_APPROX(j, jref);
}

// TODO also check actual derivatives!
template <int>
void test_autodiff_scalar()
{
  Vector2f p = Vector2f::Random();
  typedef AutoDiffScalar<Vector2f> AD;
  AD ax(p.x(),Vector2f::UnitX());
  AD ay(p.y(),Vector2f::UnitY());
  AD res = foo<AD>(ax,ay);
  VERIFY_IS_APPROX(res.value(), foo(p.x(),p.y()));
}


// TODO also check actual derivatives!
template <int>
void test_autodiff_vector()
{
  Vector2f p = Vector2f::Random();
  typedef AutoDiffScalar<Vector2f> AD;
  typedef Matrix<AD,2,1> VectorAD;
  VectorAD ap = p.cast<AD>();
  ap.x().derivatives() = Vector2f::UnitX();
  ap.y().derivatives() = Vector2f::UnitY();

  AD res = foo<VectorAD>(ap);
  VERIFY_IS_APPROX(res.value(), foo(p));
}

template <int>
void test_autodiff_jacobian()
{
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double,2,2>()) ));
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double,2,3>()) ));
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double,3,2>()) ));
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double,3,3>()) ));
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double>(3,3)) ));
#if EIGEN_HAS_VARIADIC_TEMPLATES
  CALL_SUBTEST(( forward_jacobian_cpp11(integratorFunctor<double>(10)) ));
#endif
}


template <int>
void test_autodiff_hessian()
{
  typedef AutoDiffScalar<VectorXd> AD;
  typedef Matrix<AD,Eigen::Dynamic,1> VectorAD;
  typedef AutoDiffScalar<VectorAD> ADD;
  typedef Matrix<ADD,Eigen::Dynamic,1> VectorADD;
  VectorADD x(2);
  double s1 = internal::random<double>(), s2 = internal::random<double>(), s3 = internal::random<double>(), s4 = internal::random<double>();
  x(0).value()=s1;
  x(1).value()=s2;

  //set unit vectors for the derivative directions (partial derivatives of the input vector)
  x(0).derivatives().resize(2);
  x(0).derivatives().setZero();
  x(0).derivatives()(0)= 1;
  x(1).derivatives().resize(2);
  x(1).derivatives().setZero();
  x(1).derivatives()(1)=1;

  //repeat partial derivatives for the inner AutoDiffScalar
  x(0).value().derivatives() = VectorXd::Unit(2,0);
  x(1).value().derivatives() = VectorXd::Unit(2,1);

  //set the hessian matrix to zero
  for(int idx=0; idx<2; idx++) {
      x(0).derivatives()(idx).derivatives()  = VectorXd::Zero(2);
      x(1).derivatives()(idx).derivatives()  = VectorXd::Zero(2);
  }

  ADD y = sin(AD(s3)*x(0) + AD(s4)*x(1));

  VERIFY_IS_APPROX(y.value().derivatives()(0), y.derivatives()(0).value());
  VERIFY_IS_APPROX(y.value().derivatives()(1), y.derivatives()(1).value());
  VERIFY_IS_APPROX(y.value().derivatives()(0), s3*std::cos(s1*s3+s2*s4));
  VERIFY_IS_APPROX(y.value().derivatives()(1), s4*std::cos(s1*s3+s2*s4));
  VERIFY_IS_APPROX(y.derivatives()(0).derivatives(), -std::sin(s1*s3+s2*s4)*Vector2d(s3*s3,s4*s3));
  VERIFY_IS_APPROX(y.derivatives()(1).derivatives(),  -std::sin(s1*s3+s2*s4)*Vector2d(s3*s4,s4*s4));

  ADD z = x(0)*x(1);
  VERIFY_IS_APPROX(z.derivatives()(0).derivatives(), Vector2d(0,1));
  VERIFY_IS_APPROX(z.derivatives()(1).derivatives(), Vector2d(1,0));
}

double bug_1222() {
  typedef Eigen::AutoDiffScalar<Eigen::Vector3d> AD;
  const double _cv1_3 = 1.0;
  const AD chi_3 = 1.0;
  // this line did not work, because operator+ returns ADS<DerType&>, which then cannot be converted to ADS<DerType>
  const AD denom = chi_3 + _cv1_3;
  return denom.value();
}

#ifdef EIGEN_TEST_PART_5

double bug_1223() {
  using std::min;
  typedef Eigen::AutoDiffScalar<Eigen::Vector3d> AD;

  const double _cv1_3 = 1.0;
  const AD chi_3 = 1.0;
  const AD denom = 1.0;

  // failed because implementation of min attempts to construct ADS<DerType&> via constructor AutoDiffScalar(const Real& value)
  // without initializing m_derivatives (which is a reference in this case)
  #define EIGEN_TEST_SPACE
  const AD t = min EIGEN_TEST_SPACE (denom / chi_3, 1.0);

  const AD t2 = min EIGEN_TEST_SPACE (denom / (chi_3 * _cv1_3), 1.0);

  return t.value() + t2.value();
}

// regression test for some compilation issues with specializations of ScalarBinaryOpTraits
void bug_1260() {
  Matrix4d A = Matrix4d::Ones();
  Vector4d v = Vector4d::Ones();
  A*v;
}

// check a compilation issue with numext::max
double bug_1261() {
  typedef AutoDiffScalar<Matrix2d> AD;
  typedef Matrix<AD,2,1> VectorAD;

  VectorAD v(0.,0.);
  const AD maxVal = v.maxCoeff();
  const AD minVal = v.minCoeff();
  return maxVal.value() + minVal.value();
}

double bug_1264() {
  typedef AutoDiffScalar<Vector2d> AD;
  const AD s = 0.;
  const Matrix<AD, 3, 1> v1(0.,0.,0.);
  const Matrix<AD, 3, 1> v2 = (s + 3.0) * v1;
  return v2(0).value();
}

// check with expressions on constants
double bug_1281() {
  int n = 2;
  typedef AutoDiffScalar<VectorXd> AD;
  const AD c = 1.;
  AD x0(2,n,0);
  AD y1 = (AD(c)+AD(c))*x0;
  y1 = x0 * (AD(c)+AD(c));
  AD y2 = (-AD(c))+x0;
  y2 = x0+(-AD(c));
  AD y3 = (AD(c)*(-AD(c))+AD(c))*x0;
  y3 = x0 * (AD(c)*(-AD(c))+AD(c));
  return (y1+y2+y3).value();
}

#endif

void test_autodiff()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( test_autodiff_scalar<1>() );
    CALL_SUBTEST_2( test_autodiff_vector<1>() );
    CALL_SUBTEST_3( test_autodiff_jacobian<1>() );
    CALL_SUBTEST_4( test_autodiff_hessian<1>() );
  }

  CALL_SUBTEST_5( bug_1222() );
  CALL_SUBTEST_5( bug_1223() );
  CALL_SUBTEST_5( bug_1260() );
  CALL_SUBTEST_5( bug_1261() );
  CALL_SUBTEST_5( bug_1281() );
}

