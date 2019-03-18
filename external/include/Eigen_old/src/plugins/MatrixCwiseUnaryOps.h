// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// This file is a base class plugin containing matrix specifics coefficient wise functions.

/** \returns an expression of the coefficient-wise absolute value of \c *this
  *
  * Example: \include MatrixBase_cwiseAbs.cpp
  * Output: \verbinclude MatrixBase_cwiseAbs.out
  *
  * \sa cwiseAbs2()
  */
EIGEN_STRONG_INLINE const CwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived>
cwiseAbs() const { return derived(); }

/** \returns an expression of the coefficient-wise squared absolute value of \c *this
  *
  * Example: \include MatrixBase_cwiseAbs2.cpp
  * Output: \verbinclude MatrixBase_cwiseAbs2.out
  *
  * \sa cwiseAbs()
  */
EIGEN_STRONG_INLINE const CwiseUnaryOp<internal::scalar_abs2_op<Scalar>, const Derived>
cwiseAbs2() const { return derived(); }

/** \returns an expression of the coefficient-wise square root of *this.
  *
  * Example: \include MatrixBase_cwiseSqrt.cpp
  * Output: \verbinclude MatrixBase_cwiseSqrt.out
  *
  * \sa cwisePow(), cwiseSquare()
  */
inline const CwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived>
cwiseSqrt() const { return derived(); }

/** \returns an expression of the coefficient-wise inverse of *this.
  *
  * Example: \include MatrixBase_cwiseInverse.cpp
  * Output: \verbinclude MatrixBase_cwiseInverse.out
  *
  * \sa cwiseProduct()
  */
inline const CwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived>
cwiseInverse() const { return derived(); }

