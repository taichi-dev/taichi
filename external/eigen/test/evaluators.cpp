
#include "main.h"

namespace Eigen {

  template<typename Lhs,typename Rhs>
  const Product<Lhs,Rhs>
  prod(const Lhs& lhs, const Rhs& rhs)
  {
    return Product<Lhs,Rhs>(lhs,rhs);
  }

  template<typename Lhs,typename Rhs>
  const Product<Lhs,Rhs,LazyProduct>
  lazyprod(const Lhs& lhs, const Rhs& rhs)
  {
    return Product<Lhs,Rhs,LazyProduct>(lhs,rhs);
  }
  
  template<typename DstXprType, typename SrcXprType>
  EIGEN_STRONG_INLINE
  DstXprType& copy_using_evaluator(const EigenBase<DstXprType> &dst, const SrcXprType &src)
  {
    call_assignment(dst.const_cast_derived(), src.derived(), internal::assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar>());
    return dst.const_cast_derived();
  }
  
  template<typename DstXprType, template <typename> class StorageBase, typename SrcXprType>
  EIGEN_STRONG_INLINE
  const DstXprType& copy_using_evaluator(const NoAlias<DstXprType, StorageBase>& dst, const SrcXprType &src)
  {
    call_assignment(dst, src.derived(), internal::assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar>());
    return dst.expression();
  }
  
  template<typename DstXprType, typename SrcXprType>
  EIGEN_STRONG_INLINE
  DstXprType& copy_using_evaluator(const PlainObjectBase<DstXprType> &dst, const SrcXprType &src)
  {
    #ifdef EIGEN_NO_AUTOMATIC_RESIZING
    eigen_assert((dst.size()==0 || (IsVectorAtCompileTime ? (dst.size() == src.size())
                                                          : (dst.rows() == src.rows() && dst.cols() == src.cols())))
                && "Size mismatch. Automatic resizing is disabled because EIGEN_NO_AUTOMATIC_RESIZING is defined");
  #else
    dst.const_cast_derived().resizeLike(src.derived());
  #endif
    
    call_assignment(dst.const_cast_derived(), src.derived(), internal::assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar>());
    return dst.const_cast_derived();
  }

  template<typename DstXprType, typename SrcXprType>
  void add_assign_using_evaluator(const DstXprType& dst, const SrcXprType& src)
  {
    typedef typename DstXprType::Scalar Scalar;
    call_assignment(const_cast<DstXprType&>(dst), src.derived(), internal::add_assign_op<Scalar,typename SrcXprType::Scalar>());
  }

  template<typename DstXprType, typename SrcXprType>
  void subtract_assign_using_evaluator(const DstXprType& dst, const SrcXprType& src)
  {
    typedef typename DstXprType::Scalar Scalar;
    call_assignment(const_cast<DstXprType&>(dst), src.derived(), internal::sub_assign_op<Scalar,typename SrcXprType::Scalar>());
  }

  template<typename DstXprType, typename SrcXprType>
  void multiply_assign_using_evaluator(const DstXprType& dst, const SrcXprType& src)
  {
    typedef typename DstXprType::Scalar Scalar;
    call_assignment(dst.const_cast_derived(), src.derived(), internal::mul_assign_op<Scalar,typename SrcXprType::Scalar>());
  }

  template<typename DstXprType, typename SrcXprType>
  void divide_assign_using_evaluator(const DstXprType& dst, const SrcXprType& src)
  {
    typedef typename DstXprType::Scalar Scalar;
    call_assignment(dst.const_cast_derived(), src.derived(), internal::div_assign_op<Scalar,typename SrcXprType::Scalar>());
  }
  
  template<typename DstXprType, typename SrcXprType>
  void swap_using_evaluator(const DstXprType& dst, const SrcXprType& src)
  {
    typedef typename DstXprType::Scalar Scalar;
    call_assignment(dst.const_cast_derived(), src.const_cast_derived(), internal::swap_assign_op<Scalar>());
  }

  namespace internal {
    template<typename Dst, template <typename> class StorageBase, typename Src, typename Func>
    EIGEN_DEVICE_FUNC void call_assignment(const NoAlias<Dst,StorageBase>& dst, const Src& src, const Func& func)
    {
      call_assignment_no_alias(dst.expression(), src, func);
    }
  }
  
}

template<typename XprType> long get_cost(const XprType& ) { return Eigen::internal::evaluator<XprType>::CoeffReadCost; }

using namespace std;

#define VERIFY_IS_APPROX_EVALUATOR(DEST,EXPR) VERIFY_IS_APPROX(copy_using_evaluator(DEST,(EXPR)), (EXPR).eval());
#define VERIFY_IS_APPROX_EVALUATOR2(DEST,EXPR,REF) VERIFY_IS_APPROX(copy_using_evaluator(DEST,(EXPR)), (REF).eval());

void test_evaluators()
{
  // Testing Matrix evaluator and Transpose
  Vector2d v = Vector2d::Random();
  const Vector2d v_const(v);
  Vector2d v2;
  RowVector2d w;

  VERIFY_IS_APPROX_EVALUATOR(v2, v);
  VERIFY_IS_APPROX_EVALUATOR(v2, v_const);

  // Testing Transpose
  VERIFY_IS_APPROX_EVALUATOR(w, v.transpose()); // Transpose as rvalue
  VERIFY_IS_APPROX_EVALUATOR(w, v_const.transpose());

  copy_using_evaluator(w.transpose(), v); // Transpose as lvalue
  VERIFY_IS_APPROX(w,v.transpose().eval());

  copy_using_evaluator(w.transpose(), v_const);
  VERIFY_IS_APPROX(w,v_const.transpose().eval());

  // Testing Array evaluator
  {
    ArrayXXf a(2,3);
    ArrayXXf b(3,2);
    a << 1,2,3, 4,5,6;
    const ArrayXXf a_const(a);

    VERIFY_IS_APPROX_EVALUATOR(b, a.transpose());

    VERIFY_IS_APPROX_EVALUATOR(b, a_const.transpose());

    // Testing CwiseNullaryOp evaluator
    copy_using_evaluator(w, RowVector2d::Random());
    VERIFY((w.array() >= -1).all() && (w.array() <= 1).all()); // not easy to test ...

    VERIFY_IS_APPROX_EVALUATOR(w, RowVector2d::Zero());

    VERIFY_IS_APPROX_EVALUATOR(w, RowVector2d::Constant(3));
    
    // mix CwiseNullaryOp and transpose
    VERIFY_IS_APPROX_EVALUATOR(w, Vector2d::Zero().transpose());
  }

  {
    // test product expressions
    int s = internal::random<int>(1,100);
    MatrixXf a(s,s), b(s,s), c(s,s), d(s,s);
    a.setRandom();
    b.setRandom();
    c.setRandom();
    d.setRandom();
    VERIFY_IS_APPROX_EVALUATOR(d, (a + b));
    VERIFY_IS_APPROX_EVALUATOR(d, (a + b).transpose());
    VERIFY_IS_APPROX_EVALUATOR2(d, prod(a,b), a*b);
    VERIFY_IS_APPROX_EVALUATOR2(d.noalias(), prod(a,b), a*b);
    VERIFY_IS_APPROX_EVALUATOR2(d, prod(a,b) + c, a*b + c);
    VERIFY_IS_APPROX_EVALUATOR2(d, s * prod(a,b), s * a*b);
    VERIFY_IS_APPROX_EVALUATOR2(d, prod(a,b).transpose(), (a*b).transpose());
    VERIFY_IS_APPROX_EVALUATOR2(d, prod(a,b) + prod(b,c), a*b + b*c);

    // check that prod works even with aliasing present
    c = a*a;
    copy_using_evaluator(a, prod(a,a));
    VERIFY_IS_APPROX(a,c);

    // check compound assignment of products
    d = c;
    add_assign_using_evaluator(c.noalias(), prod(a,b));
    d.noalias() += a*b;
    VERIFY_IS_APPROX(c, d);

    d = c;
    subtract_assign_using_evaluator(c.noalias(), prod(a,b));
    d.noalias() -= a*b;
    VERIFY_IS_APPROX(c, d);
  }

  {
    // test product with all possible sizes
    int s = internal::random<int>(1,100);
    Matrix<float,      1,      1> m11, res11;  m11.setRandom(1,1);
    Matrix<float,      1,      4> m14, res14;  m14.setRandom(1,4);
    Matrix<float,      1,Dynamic> m1X, res1X;  m1X.setRandom(1,s);
    Matrix<float,      4,      1> m41, res41;  m41.setRandom(4,1);
    Matrix<float,      4,      4> m44, res44;  m44.setRandom(4,4);
    Matrix<float,      4,Dynamic> m4X, res4X;  m4X.setRandom(4,s);
    Matrix<float,Dynamic,      1> mX1, resX1;  mX1.setRandom(s,1);
    Matrix<float,Dynamic,      4> mX4, resX4;  mX4.setRandom(s,4);
    Matrix<float,Dynamic,Dynamic> mXX, resXX;  mXX.setRandom(s,s);

    VERIFY_IS_APPROX_EVALUATOR2(res11, prod(m11,m11), m11*m11);
    VERIFY_IS_APPROX_EVALUATOR2(res11, prod(m14,m41), m14*m41);
    VERIFY_IS_APPROX_EVALUATOR2(res11, prod(m1X,mX1), m1X*mX1);
    VERIFY_IS_APPROX_EVALUATOR2(res14, prod(m11,m14), m11*m14);
    VERIFY_IS_APPROX_EVALUATOR2(res14, prod(m14,m44), m14*m44);
    VERIFY_IS_APPROX_EVALUATOR2(res14, prod(m1X,mX4), m1X*mX4);
    VERIFY_IS_APPROX_EVALUATOR2(res1X, prod(m11,m1X), m11*m1X);
    VERIFY_IS_APPROX_EVALUATOR2(res1X, prod(m14,m4X), m14*m4X);
    VERIFY_IS_APPROX_EVALUATOR2(res1X, prod(m1X,mXX), m1X*mXX);
    VERIFY_IS_APPROX_EVALUATOR2(res41, prod(m41,m11), m41*m11);
    VERIFY_IS_APPROX_EVALUATOR2(res41, prod(m44,m41), m44*m41);
    VERIFY_IS_APPROX_EVALUATOR2(res41, prod(m4X,mX1), m4X*mX1);
    VERIFY_IS_APPROX_EVALUATOR2(res44, prod(m41,m14), m41*m14);
    VERIFY_IS_APPROX_EVALUATOR2(res44, prod(m44,m44), m44*m44);
    VERIFY_IS_APPROX_EVALUATOR2(res44, prod(m4X,mX4), m4X*mX4);
    VERIFY_IS_APPROX_EVALUATOR2(res4X, prod(m41,m1X), m41*m1X);
    VERIFY_IS_APPROX_EVALUATOR2(res4X, prod(m44,m4X), m44*m4X);
    VERIFY_IS_APPROX_EVALUATOR2(res4X, prod(m4X,mXX), m4X*mXX);
    VERIFY_IS_APPROX_EVALUATOR2(resX1, prod(mX1,m11), mX1*m11);
    VERIFY_IS_APPROX_EVALUATOR2(resX1, prod(mX4,m41), mX4*m41);
    VERIFY_IS_APPROX_EVALUATOR2(resX1, prod(mXX,mX1), mXX*mX1);
    VERIFY_IS_APPROX_EVALUATOR2(resX4, prod(mX1,m14), mX1*m14);
    VERIFY_IS_APPROX_EVALUATOR2(resX4, prod(mX4,m44), mX4*m44);
    VERIFY_IS_APPROX_EVALUATOR2(resX4, prod(mXX,mX4), mXX*mX4);
    VERIFY_IS_APPROX_EVALUATOR2(resXX, prod(mX1,m1X), mX1*m1X);
    VERIFY_IS_APPROX_EVALUATOR2(resXX, prod(mX4,m4X), mX4*m4X);
    VERIFY_IS_APPROX_EVALUATOR2(resXX, prod(mXX,mXX), mXX*mXX);
  }

  {
    ArrayXXf a(2,3);
    ArrayXXf b(3,2);
    a << 1,2,3, 4,5,6;
    const ArrayXXf a_const(a);
    
    // this does not work because Random is eval-before-nested: 
    // copy_using_evaluator(w, Vector2d::Random().transpose());

    // test CwiseUnaryOp
    VERIFY_IS_APPROX_EVALUATOR(v2, 3 * v);
    VERIFY_IS_APPROX_EVALUATOR(w, (3 * v).transpose());
    VERIFY_IS_APPROX_EVALUATOR(b, (a + 3).transpose());
    VERIFY_IS_APPROX_EVALUATOR(b, (2 * a_const + 3).transpose());

    // test CwiseBinaryOp
    VERIFY_IS_APPROX_EVALUATOR(v2, v + Vector2d::Ones());
    VERIFY_IS_APPROX_EVALUATOR(w, (v + Vector2d::Ones()).transpose().cwiseProduct(RowVector2d::Constant(3)));

    // dynamic matrices and arrays
    MatrixXd mat1(6,6), mat2(6,6);
    VERIFY_IS_APPROX_EVALUATOR(mat1, MatrixXd::Identity(6,6));
    VERIFY_IS_APPROX_EVALUATOR(mat2, mat1);
    copy_using_evaluator(mat2.transpose(), mat1);
    VERIFY_IS_APPROX(mat2.transpose(), mat1);

    ArrayXXd arr1(6,6), arr2(6,6);
    VERIFY_IS_APPROX_EVALUATOR(arr1, ArrayXXd::Constant(6,6, 3.0));
    VERIFY_IS_APPROX_EVALUATOR(arr2, arr1);
    
    // test automatic resizing
    mat2.resize(3,3);
    VERIFY_IS_APPROX_EVALUATOR(mat2, mat1);
    arr2.resize(9,9);
    VERIFY_IS_APPROX_EVALUATOR(arr2, arr1);

    // test direct traversal
    Matrix3f m3;
    Array33f a3;
    VERIFY_IS_APPROX_EVALUATOR(m3, Matrix3f::Identity());  // matrix, nullary
    // TODO: find a way to test direct traversal with array
    VERIFY_IS_APPROX_EVALUATOR(m3.transpose(), Matrix3f::Identity().transpose());  // transpose
    VERIFY_IS_APPROX_EVALUATOR(m3, 2 * Matrix3f::Identity());  // unary
    VERIFY_IS_APPROX_EVALUATOR(m3, Matrix3f::Identity() + Matrix3f::Zero());  // binary
    VERIFY_IS_APPROX_EVALUATOR(m3.block(0,0,2,2), Matrix3f::Identity().block(1,1,2,2));  // block

    // test linear traversal
    VERIFY_IS_APPROX_EVALUATOR(m3, Matrix3f::Zero());  // matrix, nullary
    VERIFY_IS_APPROX_EVALUATOR(a3, Array33f::Zero());  // array
    VERIFY_IS_APPROX_EVALUATOR(m3.transpose(), Matrix3f::Zero().transpose());  // transpose
    VERIFY_IS_APPROX_EVALUATOR(m3, 2 * Matrix3f::Zero());  // unary
    VERIFY_IS_APPROX_EVALUATOR(m3, Matrix3f::Zero() + m3);  // binary  

    // test inner vectorization
    Matrix4f m4, m4src = Matrix4f::Random();
    Array44f a4, a4src = Matrix4f::Random();
    VERIFY_IS_APPROX_EVALUATOR(m4, m4src);  // matrix
    VERIFY_IS_APPROX_EVALUATOR(a4, a4src);  // array
    VERIFY_IS_APPROX_EVALUATOR(m4.transpose(), m4src.transpose());  // transpose
    // TODO: find out why Matrix4f::Zero() does not allow inner vectorization
    VERIFY_IS_APPROX_EVALUATOR(m4, 2 * m4src);  // unary
    VERIFY_IS_APPROX_EVALUATOR(m4, m4src + m4src);  // binary

    // test linear vectorization
    MatrixXf mX(6,6), mXsrc = MatrixXf::Random(6,6);
    ArrayXXf aX(6,6), aXsrc = ArrayXXf::Random(6,6);
    VERIFY_IS_APPROX_EVALUATOR(mX, mXsrc);  // matrix
    VERIFY_IS_APPROX_EVALUATOR(aX, aXsrc);  // array
    VERIFY_IS_APPROX_EVALUATOR(mX.transpose(), mXsrc.transpose());  // transpose
    VERIFY_IS_APPROX_EVALUATOR(mX, MatrixXf::Zero(6,6));  // nullary
    VERIFY_IS_APPROX_EVALUATOR(mX, 2 * mXsrc);  // unary
    VERIFY_IS_APPROX_EVALUATOR(mX, mXsrc + mXsrc);  // binary

    // test blocks and slice vectorization
    VERIFY_IS_APPROX_EVALUATOR(m4, (mXsrc.block<4,4>(1,0)));
    VERIFY_IS_APPROX_EVALUATOR(aX, ArrayXXf::Constant(10, 10, 3.0).block(2, 3, 6, 6));

    Matrix4f m4ref = m4;
    copy_using_evaluator(m4.block(1, 1, 2, 3), m3.bottomRows(2));
    m4ref.block(1, 1, 2, 3) = m3.bottomRows(2);
    VERIFY_IS_APPROX(m4, m4ref);

    mX.setIdentity(20,20);
    MatrixXf mXref = MatrixXf::Identity(20,20);
    mXsrc = MatrixXf::Random(9,12);
    copy_using_evaluator(mX.block(4, 4, 9, 12), mXsrc);
    mXref.block(4, 4, 9, 12) = mXsrc;
    VERIFY_IS_APPROX(mX, mXref);

    // test Map
    const float raw[3] = {1,2,3};
    float buffer[3] = {0,0,0};
    Vector3f v3;
    Array3f a3f;
    VERIFY_IS_APPROX_EVALUATOR(v3, Map<const Vector3f>(raw));
    VERIFY_IS_APPROX_EVALUATOR(a3f, Map<const Array3f>(raw));
    Vector3f::Map(buffer) = 2*v3;
    VERIFY(buffer[0] == 2);
    VERIFY(buffer[1] == 4);
    VERIFY(buffer[2] == 6);

    // test CwiseUnaryView
    mat1.setRandom();
    mat2.setIdentity();
    MatrixXcd matXcd(6,6), matXcd_ref(6,6);
    copy_using_evaluator(matXcd.real(), mat1);
    copy_using_evaluator(matXcd.imag(), mat2);
    matXcd_ref.real() = mat1;
    matXcd_ref.imag() = mat2;
    VERIFY_IS_APPROX(matXcd, matXcd_ref);

    // test Select
    VERIFY_IS_APPROX_EVALUATOR(aX, (aXsrc > 0).select(aXsrc, -aXsrc));

    // test Replicate
    mXsrc = MatrixXf::Random(6, 6);
    VectorXf vX = VectorXf::Random(6);
    mX.resize(6, 6);
    VERIFY_IS_APPROX_EVALUATOR(mX, mXsrc.colwise() + vX);
    matXcd.resize(12, 12);
    VERIFY_IS_APPROX_EVALUATOR(matXcd, matXcd_ref.replicate(2,2));
    VERIFY_IS_APPROX_EVALUATOR(matXcd, (matXcd_ref.replicate<2,2>()));

    // test partial reductions
    VectorXd vec1(6);
    VERIFY_IS_APPROX_EVALUATOR(vec1, mat1.rowwise().sum());
    VERIFY_IS_APPROX_EVALUATOR(vec1, mat1.colwise().sum().transpose());

    // test MatrixWrapper and ArrayWrapper
    mat1.setRandom(6,6);
    arr1.setRandom(6,6);
    VERIFY_IS_APPROX_EVALUATOR(mat2, arr1.matrix());
    VERIFY_IS_APPROX_EVALUATOR(arr2, mat1.array());
    VERIFY_IS_APPROX_EVALUATOR(mat2, (arr1 + 2).matrix());
    VERIFY_IS_APPROX_EVALUATOR(arr2, mat1.array() + 2);
    mat2.array() = arr1 * arr1;
    VERIFY_IS_APPROX(mat2, (arr1 * arr1).matrix());
    arr2.matrix() = MatrixXd::Identity(6,6);
    VERIFY_IS_APPROX(arr2, MatrixXd::Identity(6,6).array());

    // test Reverse
    VERIFY_IS_APPROX_EVALUATOR(arr2, arr1.reverse());
    VERIFY_IS_APPROX_EVALUATOR(arr2, arr1.colwise().reverse());
    VERIFY_IS_APPROX_EVALUATOR(arr2, arr1.rowwise().reverse());
    arr2.reverse() = arr1;
    VERIFY_IS_APPROX(arr2, arr1.reverse());
    mat2.array() = mat1.array().reverse();
    VERIFY_IS_APPROX(mat2.array(), mat1.array().reverse());

    // test Diagonal
    VERIFY_IS_APPROX_EVALUATOR(vec1, mat1.diagonal());
    vec1.resize(5);
    VERIFY_IS_APPROX_EVALUATOR(vec1, mat1.diagonal(1));
    VERIFY_IS_APPROX_EVALUATOR(vec1, mat1.diagonal<-1>());
    vec1.setRandom();

    mat2 = mat1;
    copy_using_evaluator(mat1.diagonal(1), vec1);
    mat2.diagonal(1) = vec1;
    VERIFY_IS_APPROX(mat1, mat2);

    copy_using_evaluator(mat1.diagonal<-1>(), mat1.diagonal(1));
    mat2.diagonal<-1>() = mat2.diagonal(1);
    VERIFY_IS_APPROX(mat1, mat2);
  }
  
  {
    // test swapping
    MatrixXd mat1, mat2, mat1ref, mat2ref;
    mat1ref = mat1 = MatrixXd::Random(6, 6);
    mat2ref = mat2 = 2 * mat1 + MatrixXd::Identity(6, 6);
    swap_using_evaluator(mat1, mat2);
    mat1ref.swap(mat2ref);
    VERIFY_IS_APPROX(mat1, mat1ref);
    VERIFY_IS_APPROX(mat2, mat2ref);

    swap_using_evaluator(mat1.block(0, 0, 3, 3), mat2.block(3, 3, 3, 3));
    mat1ref.block(0, 0, 3, 3).swap(mat2ref.block(3, 3, 3, 3));
    VERIFY_IS_APPROX(mat1, mat1ref);
    VERIFY_IS_APPROX(mat2, mat2ref);

    swap_using_evaluator(mat1.row(2), mat2.col(3).transpose());
    mat1.row(2).swap(mat2.col(3).transpose());
    VERIFY_IS_APPROX(mat1, mat1ref);
    VERIFY_IS_APPROX(mat2, mat2ref);
  }

  {
    // test compound assignment
    const Matrix4d mat_const = Matrix4d::Random(); 
    Matrix4d mat, mat_ref;
    mat = mat_ref = Matrix4d::Identity();
    add_assign_using_evaluator(mat, mat_const);
    mat_ref += mat_const;
    VERIFY_IS_APPROX(mat, mat_ref);

    subtract_assign_using_evaluator(mat.row(1), 2*mat.row(2));
    mat_ref.row(1) -= 2*mat_ref.row(2);
    VERIFY_IS_APPROX(mat, mat_ref);

    const ArrayXXf arr_const = ArrayXXf::Random(5,3); 
    ArrayXXf arr, arr_ref;
    arr = arr_ref = ArrayXXf::Constant(5, 3, 0.5);
    multiply_assign_using_evaluator(arr, arr_const);
    arr_ref *= arr_const;
    VERIFY_IS_APPROX(arr, arr_ref);

    divide_assign_using_evaluator(arr.row(1), arr.row(2) + 1);
    arr_ref.row(1) /= (arr_ref.row(2) + 1);
    VERIFY_IS_APPROX(arr, arr_ref);
  }
  
  {
    // test triangular shapes
    MatrixXd A = MatrixXd::Random(6,6), B(6,6), C(6,6), D(6,6);
    A.setRandom();B.setRandom();
    VERIFY_IS_APPROX_EVALUATOR2(B, A.triangularView<Upper>(), MatrixXd(A.triangularView<Upper>()));
    
    A.setRandom();B.setRandom();
    VERIFY_IS_APPROX_EVALUATOR2(B, A.triangularView<UnitLower>(), MatrixXd(A.triangularView<UnitLower>()));
    
    A.setRandom();B.setRandom();
    VERIFY_IS_APPROX_EVALUATOR2(B, A.triangularView<UnitUpper>(), MatrixXd(A.triangularView<UnitUpper>()));
    
    A.setRandom();B.setRandom();
    C = B; C.triangularView<Upper>() = A;
    copy_using_evaluator(B.triangularView<Upper>(), A);
    VERIFY(B.isApprox(C) && "copy_using_evaluator(B.triangularView<Upper>(), A)");
    
    A.setRandom();B.setRandom();
    C = B; C.triangularView<Lower>() = A.triangularView<Lower>();
    copy_using_evaluator(B.triangularView<Lower>(), A.triangularView<Lower>());
    VERIFY(B.isApprox(C) && "copy_using_evaluator(B.triangularView<Lower>(), A.triangularView<Lower>())");
    
    
    A.setRandom();B.setRandom();
    C = B; C.triangularView<Lower>() = A.triangularView<Upper>().transpose();
    copy_using_evaluator(B.triangularView<Lower>(), A.triangularView<Upper>().transpose());
    VERIFY(B.isApprox(C) && "copy_using_evaluator(B.triangularView<Lower>(), A.triangularView<Lower>().transpose())");
    
    
    A.setRandom();B.setRandom(); C = B; D = A;
    C.triangularView<Upper>().swap(D.triangularView<Upper>());
    swap_using_evaluator(B.triangularView<Upper>(), A.triangularView<Upper>());
    VERIFY(B.isApprox(C) && "swap_using_evaluator(B.triangularView<Upper>(), A.triangularView<Upper>())");
    
    
    VERIFY_IS_APPROX_EVALUATOR2(B, prod(A.triangularView<Upper>(),A), MatrixXd(A.triangularView<Upper>()*A));
    
    VERIFY_IS_APPROX_EVALUATOR2(B, prod(A.selfadjointView<Upper>(),A), MatrixXd(A.selfadjointView<Upper>()*A));
  }

  {
    // test diagonal shapes
    VectorXd d = VectorXd::Random(6);
    MatrixXd A = MatrixXd::Random(6,6), B(6,6);
    A.setRandom();B.setRandom();
    
    VERIFY_IS_APPROX_EVALUATOR2(B, lazyprod(d.asDiagonal(),A), MatrixXd(d.asDiagonal()*A));
    VERIFY_IS_APPROX_EVALUATOR2(B, lazyprod(A,d.asDiagonal()), MatrixXd(A*d.asDiagonal()));
  }

  {
    // test CoeffReadCost
    Matrix4d a, b;
    VERIFY_IS_EQUAL( get_cost(a), 1 );
    VERIFY_IS_EQUAL( get_cost(a+b), 3);
    VERIFY_IS_EQUAL( get_cost(2*a+b), 4);
    VERIFY_IS_EQUAL( get_cost(a*b), 1);
    VERIFY_IS_EQUAL( get_cost(a.lazyProduct(b)), 15);
    VERIFY_IS_EQUAL( get_cost(a*(a*b)), 1);
    VERIFY_IS_EQUAL( get_cost(a.lazyProduct(a*b)), 15);
    VERIFY_IS_EQUAL( get_cost(a*(a+b)), 1);
    VERIFY_IS_EQUAL( get_cost(a.lazyProduct(a+b)), 15);
  }
}
