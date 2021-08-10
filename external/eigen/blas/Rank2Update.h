// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_RANK2UPDATE_H
#define EIGEN_RANK2UPDATE_H

namespace internal {

/* Optimized selfadjoint matrix += alpha * uv' + conj(alpha)*vu'
 * This is the low-level version of SelfadjointRank2Update.h
 */
template<typename Scalar, typename Index, int UpLo>
struct rank2_update_selector
{
  static void run(Index size, Scalar* mat, Index stride, const Scalar* u, const Scalar* v, Scalar alpha)
  {
    typedef Map<const Matrix<Scalar,Dynamic,1> > OtherMap;
    for (Index i=0; i<size; ++i)
    {
      Map<Matrix<Scalar,Dynamic,1> >(mat+stride*i+(UpLo==Lower ? i : 0), UpLo==Lower ? size-i : (i+1)) +=
      numext::conj(alpha) * numext::conj(u[i]) * OtherMap(v+(UpLo==Lower ? i : 0), UpLo==Lower ? size-i : (i+1))
                + alpha * numext::conj(v[i]) * OtherMap(u+(UpLo==Lower ? i : 0), UpLo==Lower ? size-i : (i+1));
    }
  }
};

/* Optimized selfadjoint matrix += alpha * uv' + conj(alpha)*vu'
 * The matrix is in packed form.
 */
template<typename Scalar, typename Index, int UpLo>
struct packed_rank2_update_selector
{
  static void run(Index size, Scalar* mat, const Scalar* u, const Scalar* v, Scalar alpha)
  {
    typedef Map<const Matrix<Scalar,Dynamic,1> > OtherMap;
    Index offset = 0;
    for (Index i=0; i<size; ++i)
    {
      Map<Matrix<Scalar,Dynamic,1> >(mat+offset, UpLo==Lower ? size-i : (i+1)) +=
      numext::conj(alpha) * numext::conj(u[i]) * OtherMap(v+(UpLo==Lower ? i : 0), UpLo==Lower ? size-i : (i+1))
                + alpha * numext::conj(v[i]) * OtherMap(u+(UpLo==Lower ? i : 0), UpLo==Lower ? size-i : (i+1));
      //FIXME This should be handled outside.
      mat[offset+(UpLo==Lower ? 0 : i)] = numext::real(mat[offset+(UpLo==Lower ? 0 : i)]);
      offset += UpLo==Lower ? size-i : (i+1);
    }
  }
};

} // end namespace internal

#endif // EIGEN_RANK2UPDATE_H
