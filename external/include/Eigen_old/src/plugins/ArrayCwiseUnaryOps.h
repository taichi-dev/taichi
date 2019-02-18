

/** \returns an expression of the coefficient-wise absolute value of \c *this
  *
  * Example: \include Cwise_abs.cpp
  * Output: \verbinclude Cwise_abs.out
  *
  * \sa abs2()
  */
EIGEN_STRONG_INLINE const CwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived>
abs() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise squared absolute value of \c *this
  *
  * Example: \include Cwise_abs2.cpp
  * Output: \verbinclude Cwise_abs2.out
  *
  * \sa abs(), square()
  */
EIGEN_STRONG_INLINE const CwiseUnaryOp<internal::scalar_abs2_op<Scalar>, const Derived>
abs2() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise exponential of *this.
  *
  * Example: \include Cwise_exp.cpp
  * Output: \verbinclude Cwise_exp.out
  *
  * \sa pow(), log(), sin(), cos()
  */
inline const CwiseUnaryOp<internal::scalar_exp_op<Scalar>, const Derived>
exp() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise logarithm of *this.
  *
  * Example: \include Cwise_log.cpp
  * Output: \verbinclude Cwise_log.out
  *
  * \sa exp()
  */
inline const CwiseUnaryOp<internal::scalar_log_op<Scalar>, const Derived>
log() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise square root of *this.
  *
  * Example: \include Cwise_sqrt.cpp
  * Output: \verbinclude Cwise_sqrt.out
  *
  * \sa pow(), square()
  */
inline const CwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived>
sqrt() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise cosine of *this.
  *
  * Example: \include Cwise_cos.cpp
  * Output: \verbinclude Cwise_cos.out
  *
  * \sa sin(), acos()
  */
inline const CwiseUnaryOp<internal::scalar_cos_op<Scalar>, const Derived>
cos() const
{
  return derived();
}


/** \returns an expression of the coefficient-wise sine of *this.
  *
  * Example: \include Cwise_sin.cpp
  * Output: \verbinclude Cwise_sin.out
  *
  * \sa cos(), asin()
  */
inline const CwiseUnaryOp<internal::scalar_sin_op<Scalar>, const Derived>
sin() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise arc cosine of *this.
  *
  * Example: \include Cwise_acos.cpp
  * Output: \verbinclude Cwise_acos.out
  *
  * \sa cos(), asin()
  */
inline const CwiseUnaryOp<internal::scalar_acos_op<Scalar>, const Derived>
acos() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise arc sine of *this.
  *
  * Example: \include Cwise_asin.cpp
  * Output: \verbinclude Cwise_asin.out
  *
  * \sa sin(), acos()
  */
inline const CwiseUnaryOp<internal::scalar_asin_op<Scalar>, const Derived>
asin() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise tan of *this.
  *
  * Example: \include Cwise_tan.cpp
  * Output: \verbinclude Cwise_tan.out
  *
  * \sa cos(), sin()
  */
inline const CwiseUnaryOp<internal::scalar_tan_op<Scalar>, Derived>
tan() const
{
  return derived();
}


/** \returns an expression of the coefficient-wise power of *this to the given exponent.
  *
  * Example: \include Cwise_pow.cpp
  * Output: \verbinclude Cwise_pow.out
  *
  * \sa exp(), log()
  */
inline const CwiseUnaryOp<internal::scalar_pow_op<Scalar>, const Derived>
pow(const Scalar& exponent) const
{
  return CwiseUnaryOp<internal::scalar_pow_op<Scalar>, const Derived>
          (derived(), internal::scalar_pow_op<Scalar>(exponent));
}


/** \returns an expression of the coefficient-wise inverse of *this.
  *
  * Example: \include Cwise_inverse.cpp
  * Output: \verbinclude Cwise_inverse.out
  *
  * \sa operator/(), operator*()
  */
inline const CwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived>
inverse() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise square of *this.
  *
  * Example: \include Cwise_square.cpp
  * Output: \verbinclude Cwise_square.out
  *
  * \sa operator/(), operator*(), abs2()
  */
inline const CwiseUnaryOp<internal::scalar_square_op<Scalar>, const Derived>
square() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise cube of *this.
  *
  * Example: \include Cwise_cube.cpp
  * Output: \verbinclude Cwise_cube.out
  *
  * \sa square(), pow()
  */
inline const CwiseUnaryOp<internal::scalar_cube_op<Scalar>, const Derived>
cube() const
{
  return derived();
}
