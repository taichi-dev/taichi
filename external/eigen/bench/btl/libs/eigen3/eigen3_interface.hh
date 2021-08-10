//=====================================================
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//=====================================================
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
#ifndef EIGEN3_INTERFACE_HH
#define EIGEN3_INTERFACE_HH

#include <Eigen/Eigen>
#include <vector>
#include "btl.hh"

using namespace Eigen;

template<class real, int SIZE=Dynamic>
class eigen3_interface
{

public :

  enum {IsFixedSize = (SIZE!=Dynamic)};

  typedef real real_type;

  typedef std::vector<real> stl_vector;
  typedef std::vector<stl_vector> stl_matrix;

  typedef Eigen::Matrix<real,SIZE,SIZE> gene_matrix;
  typedef Eigen::Matrix<real,SIZE,1> gene_vector;

  static inline std::string name( void )
  {
    return EIGEN_MAKESTRING(BTL_PREFIX);
  }

  static void free_matrix(gene_matrix & /*A*/, int /*N*/) {}

  static void free_vector(gene_vector & /*B*/) {}

  static BTL_DONT_INLINE void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    A.resize(A_stl[0].size(), A_stl.size());

    for (unsigned int j=0; j<A_stl.size() ; j++){
      for (unsigned int i=0; i<A_stl[j].size() ; i++){
        A.coeffRef(i,j) = A_stl[j][i];
      }
    }
  }

  static BTL_DONT_INLINE  void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    B.resize(B_stl.size(),1);

    for (unsigned int i=0; i<B_stl.size() ; i++){
      B.coeffRef(i) = B_stl[i];
    }
  }

  static BTL_DONT_INLINE  void vector_to_stl(gene_vector & B, stl_vector & B_stl){
    for (unsigned int i=0; i<B_stl.size() ; i++){
      B_stl[i] = B.coeff(i);
    }
  }

  static BTL_DONT_INLINE  void matrix_to_stl(gene_matrix & A, stl_matrix & A_stl){
    int  N=A_stl.size();

    for (int j=0;j<N;j++){
      A_stl[j].resize(N);
      for (int i=0;i<N;i++){
        A_stl[j][i] = A.coeff(i,j);
      }
    }
  }

  static inline void matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int  /*N*/){
    X.noalias() = A*B;
  }

  static inline void transposed_matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int  /*N*/){
    X.noalias() = A.transpose()*B.transpose();
  }

//   static inline void ata_product(const gene_matrix & A, gene_matrix & X, int  /*N*/){
//     X.noalias() = A.transpose()*A;
//   }

  static inline void aat_product(const gene_matrix & A, gene_matrix & X, int  /*N*/){
    X.template triangularView<Lower>().setZero();
    X.template selfadjointView<Lower>().rankUpdate(A);
  }

  static inline void matrix_vector_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int  /*N*/){
    X.noalias() = A*B;
  }

  static inline void symv(const gene_matrix & A, const gene_vector & B, gene_vector & X, int  /*N*/){
    X.noalias() = (A.template selfadjointView<Lower>() * B);
//     internal::product_selfadjoint_vector<real,0,LowerTriangularBit,false,false>(N,A.data(),N, B.data(), 1, X.data(), 1);
  }

  template<typename Dest, typename Src> static void triassign(Dest& dst, const Src& src)
  {
    typedef typename Dest::Scalar Scalar;
    typedef typename internal::packet_traits<Scalar>::type Packet;
    const int PacketSize = sizeof(Packet)/sizeof(Scalar);
    int size = dst.cols();
    for(int j=0; j<size; j+=1)
    {
//       const int alignedEnd = alignedStart + ((innerSize-alignedStart) & ~packetAlignedMask);
      Scalar* A0 = dst.data() + j*dst.stride();
      int starti = j;
      int alignedEnd = starti;
      int alignedStart = (starti) + internal::first_aligned(&A0[starti], size-starti);
      alignedEnd = alignedStart + ((size-alignedStart)/(2*PacketSize))*(PacketSize*2);

      // do the non-vectorizable part of the assignment
      for (int index = starti; index<alignedStart ; ++index)
      {
        if(Dest::Flags&RowMajorBit)
          dst.copyCoeff(j, index, src);
        else
          dst.copyCoeff(index, j, src);
      }

      // do the vectorizable part of the assignment
      for (int index = alignedStart; index<alignedEnd; index+=PacketSize)
      {
        if(Dest::Flags&RowMajorBit)
          dst.template copyPacket<Src, Aligned, Unaligned>(j, index, src);
        else
          dst.template copyPacket<Src, Aligned, Unaligned>(index, j, src);
      }

      // do the non-vectorizable part of the assignment
      for (int index = alignedEnd; index<size; ++index)
      {
        if(Dest::Flags&RowMajorBit)
          dst.copyCoeff(j, index, src);
        else
          dst.copyCoeff(index, j, src);
      }
      //dst.col(j).tail(N-j) = src.col(j).tail(N-j);
    }
  }

  static EIGEN_DONT_INLINE void syr2(gene_matrix & A,  gene_vector & X, gene_vector & Y, int  N){
    // internal::product_selfadjoint_rank2_update<real,0,LowerTriangularBit>(N,A.data(),N, X.data(), 1, Y.data(), 1, -1);
    for(int j=0; j<N; ++j)
      A.col(j).tail(N-j) += X[j] * Y.tail(N-j) + Y[j] * X.tail(N-j);
  }

  static EIGEN_DONT_INLINE void ger(gene_matrix & A,  gene_vector & X, gene_vector & Y, int  N){
    for(int j=0; j<N; ++j)
      A.col(j) += X * Y[j];
  }

  static EIGEN_DONT_INLINE void rot(gene_vector & A,  gene_vector & B, real c, real s, int  /*N*/){
    internal::apply_rotation_in_the_plane(A, B, JacobiRotation<real>(c,s));
  }

  static inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int  /*N*/){
    X.noalias() = (A.transpose()*B);
  }

  static inline void axpy(real coef, const gene_vector & X, gene_vector & Y, int  /*N*/){
    Y += coef * X;
  }

  static inline void axpby(real a, const gene_vector & X, real b, gene_vector & Y, int  /*N*/){
    Y = a*X + b*Y;
  }

  static EIGEN_DONT_INLINE void copy_matrix(const gene_matrix & source, gene_matrix & cible, int  /*N*/){
    cible = source;
  }

  static EIGEN_DONT_INLINE void copy_vector(const gene_vector & source, gene_vector & cible, int  /*N*/){
    cible = source;
  }

  static inline void trisolve_lower(const gene_matrix & L, const gene_vector& B, gene_vector& X, int  /*N*/){
    X = L.template triangularView<Lower>().solve(B);
  }

  static inline void trisolve_lower_matrix(const gene_matrix & L, const gene_matrix& B, gene_matrix& X, int  /*N*/){
    X = L.template triangularView<Upper>().solve(B);
  }

  static inline void trmm(const gene_matrix & L, const gene_matrix& B, gene_matrix& X, int  /*N*/){
    X.noalias() = L.template triangularView<Lower>() * B;
  }

  static inline void cholesky(const gene_matrix & X, gene_matrix & C, int  /*N*/){
    C = X;
    internal::llt_inplace<real,Lower>::blocked(C);
    //C = X.llt().matrixL();
//     C = X;
//     Cholesky<gene_matrix>::computeInPlace(C);
//     Cholesky<gene_matrix>::computeInPlaceBlock(C);
  }

  static inline void lu_decomp(const gene_matrix & X, gene_matrix & C, int  /*N*/){
    C = X.fullPivLu().matrixLU();
  }

  static inline void partial_lu_decomp(const gene_matrix & X, gene_matrix & C, int  N){
    Matrix<DenseIndex,1,Dynamic> piv(N);
    DenseIndex nb;
    C = X;
    internal::partial_lu_inplace(C,piv,nb);
//     C = X.partialPivLu().matrixLU();
  }

  static inline void tridiagonalization(const gene_matrix & X, gene_matrix & C, int  N){
    typename Tridiagonalization<gene_matrix>::CoeffVectorType aux(N-1);
    C = X;
    internal::tridiagonalization_inplace(C, aux);
  }

  static inline void hessenberg(const gene_matrix & X, gene_matrix & C, int  /*N*/){
    C = HessenbergDecomposition<gene_matrix>(X).packedMatrix();
  }



};

#endif
