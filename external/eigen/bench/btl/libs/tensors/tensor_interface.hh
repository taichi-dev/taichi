//=====================================================
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//=====================================================
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
#ifndef TENSOR_INTERFACE_HH
#define TENSOR_INTERFACE_HH

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include "btl.hh"

using namespace Eigen;

template<class real>
class tensor_interface
{
public :
  typedef real real_type;
  typedef typename Eigen::Tensor<real,2>::Index Index;

  typedef std::vector<real> stl_vector;
  typedef std::vector<stl_vector> stl_matrix;

  typedef Eigen::Tensor<real,2> gene_matrix;
  typedef Eigen::Tensor<real,1> gene_vector;


  static inline std::string name( void )
  {
    return EIGEN_MAKESTRING(BTL_PREFIX);
  }

  static void free_matrix(gene_matrix & /*A*/, int /*N*/) {}

  static void free_vector(gene_vector & /*B*/) {}

  static BTL_DONT_INLINE void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    A.resize(Eigen::array<Index,2>(A_stl[0].size(), A_stl.size()));

    for (unsigned int j=0; j<A_stl.size() ; j++){
      for (unsigned int i=0; i<A_stl[j].size() ; i++){
        A.coeffRef(Eigen::array<Index,2>(i,j)) = A_stl[j][i];
      }
    }
  }

  static BTL_DONT_INLINE  void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    B.resize(B_stl.size());

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
        A_stl[j][i] = A.coeff(Eigen::array<Index,2>(i,j));
      }
    }
  }

  static inline void matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int  /*N*/){
    typedef typename Eigen::Tensor<real_type, 1>::DimensionPair DimPair;
    const Eigen::array<DimPair, 1> dims(DimPair(1, 0));
    X/*.noalias()*/ = A.contract(B, dims);
  }

  static inline void matrix_vector_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int  /*N*/){
    typedef typename Eigen::Tensor<real_type, 1>::DimensionPair DimPair;
    const Eigen::array<DimPair, 1> dims(DimPair(1, 0));
    X/*.noalias()*/ = A.contract(B, dims);
  }

  static inline void axpy(real coef, const gene_vector & X, gene_vector & Y, int  /*N*/){
    Y += X.constant(coef) * X;
  }

  static inline void axpby(real a, const gene_vector & X, real b, gene_vector & Y, int  /*N*/){
    Y = X.constant(a)*X + Y.constant(b)*Y;
  }

  static EIGEN_DONT_INLINE void copy_matrix(const gene_matrix & source, gene_matrix & cible, int  /*N*/){
    cible = source;
  }

  static EIGEN_DONT_INLINE void copy_vector(const gene_vector & source, gene_vector & cible, int  /*N*/){
    cible = source;
  }
};

#endif
