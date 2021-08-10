
#include <iostream>
#include <Eigen/Geometry>
#include <bench/BenchTimer.h>

using namespace std;
using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

#ifndef SIZE
#define SIZE 8
#endif

typedef SCALAR Scalar;
typedef NumTraits<Scalar>::Real RealScalar;
typedef Matrix<RealScalar,Dynamic,Dynamic> A;
typedef Matrix</*Real*/Scalar,Dynamic,Dynamic> B;
typedef Matrix<Scalar,Dynamic,Dynamic> C;
typedef Matrix<RealScalar,Dynamic,Dynamic> M;

template<typename Transformation, typename Data>
EIGEN_DONT_INLINE void transform(const Transformation& t, Data& data)
{
  EIGEN_ASM_COMMENT("begin");
  data = t * data;
  EIGEN_ASM_COMMENT("end");
}

template<typename Scalar, typename Data>
EIGEN_DONT_INLINE void transform(const Quaternion<Scalar>& t, Data& data)
{
  EIGEN_ASM_COMMENT("begin quat");
  for(int i=0;i<data.cols();++i)
    data.col(i) = t * data.col(i);
  EIGEN_ASM_COMMENT("end quat");
}

template<typename T> struct ToRotationMatrixWrapper
{
  enum {Dim = T::Dim};
  typedef typename T::Scalar Scalar;
  ToRotationMatrixWrapper(const T& o) : object(o) {}
  T object;
};

template<typename QType, typename Data>
EIGEN_DONT_INLINE void transform(const ToRotationMatrixWrapper<QType>& t, Data& data)
{
  EIGEN_ASM_COMMENT("begin quat via mat");
  data = t.object.toRotationMatrix() * data;
  EIGEN_ASM_COMMENT("end quat via mat");
}

template<typename Scalar, int Dim, typename Data>
EIGEN_DONT_INLINE void transform(const Transform<Scalar,Dim,Projective>& t, Data& data)
{
  data = (t * data.colwise().homogeneous()).template block<Dim,Data::ColsAtCompileTime>(0,0);
}

template<typename T> struct get_dim { enum { Dim = T::Dim }; };
template<typename S, int R, int C, int O, int MR, int MC>
struct get_dim<Matrix<S,R,C,O,MR,MC> > { enum { Dim = R }; };

template<typename Transformation, int N>
struct bench_impl
{
  static EIGEN_DONT_INLINE void run(const Transformation& t)
  {
    Matrix<typename Transformation::Scalar,get_dim<Transformation>::Dim,N> data;
    data.setRandom();
    bench_impl<Transformation,N-1>::run(t);
    BenchTimer timer;
    BENCH(timer,10,100000,transform(t,data));
    cout.width(9);
    cout << timer.best() << " ";
  }
};


template<typename Transformation>
struct bench_impl<Transformation,0>
{
  static EIGEN_DONT_INLINE void run(const Transformation&) {}
};

template<typename Transformation>
EIGEN_DONT_INLINE void bench(const std::string& msg, const Transformation& t)
{
  cout << msg << " ";
  bench_impl<Transformation,SIZE>::run(t);
  std::cout << "\n";
}

int main(int argc, char ** argv)
{
  Matrix<Scalar,3,4> mat34; mat34.setRandom();
  Transform<Scalar,3,Isometry> iso3(mat34);
  Transform<Scalar,3,Affine> aff3(mat34);
  Transform<Scalar,3,AffineCompact> caff3(mat34);
  Transform<Scalar,3,Projective> proj3(mat34);
  Quaternion<Scalar> quat;quat.setIdentity();
  ToRotationMatrixWrapper<Quaternion<Scalar> > quatmat(quat);
  Matrix<Scalar,3,3> mat33; mat33.setRandom();
  
  cout.precision(4);
  std::cout
     << "N          ";
  for(int i=0;i<SIZE;++i)
  {
    cout.width(9);
    cout << i+1 << " ";
  }
  cout << "\n";
  
  bench("matrix 3x3", mat33);
  bench("quaternion", quat);
  bench("quat-mat  ", quatmat);
  bench("isometry3 ", iso3);
  bench("affine3   ", aff3);
  bench("c affine3 ", caff3);
  bench("proj3     ", proj3);
}

