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
#ifndef GMM_INTERFACE_HH
#define GMM_INTERFACE_HH

#include <gmm/gmm.h>
#include <vector>

using namespace gmm;

template<class real>
class gmm_interface {

public :

  typedef real real_type ;

  typedef std::vector<real>  stl_vector;
  typedef std::vector<stl_vector > stl_matrix;

  typedef gmm::dense_matrix<real> gene_matrix;
  typedef stl_vector gene_vector;

  static inline std::string name( void )
  {
    return "gmm";
  }

  static void free_matrix(gene_matrix & A, int N){
    return ;
  }

  static void free_vector(gene_vector & B){
    return ;
  }

  static inline void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    A.resize(A_stl[0].size(),A_stl.size());

    for (int j=0; j<A_stl.size() ; j++){
      for (int i=0; i<A_stl[j].size() ; i++){
        A(i,j) = A_stl[j][i];
      }
    }
  }

  static inline void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    B = B_stl;
  }

  static inline void vector_to_stl(gene_vector & B, stl_vector & B_stl){
    B_stl = B;
  }

  static inline void matrix_to_stl(gene_matrix & A, stl_matrix & A_stl){
    int N=A_stl.size();

    for (int j=0;j<N;j++){
      A_stl[j].resize(N);
      for (int i=0;i<N;i++){
        A_stl[j][i] = A(i,j);
      }
    }
  }

  static inline void matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int N){
    gmm::mult(A,B, X);
  }

  static inline void transposed_matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int N){
    gmm::mult(gmm::transposed(A),gmm::transposed(B), X);
  }

  static inline void ata_product(const gene_matrix & A, gene_matrix & X, int N){
    gmm::mult(gmm::transposed(A),A, X);
  }

  static inline void aat_product(const gene_matrix & A, gene_matrix & X, int N){
    gmm::mult(A,gmm::transposed(A), X);
  }

  static inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    gmm::mult(A,B,X);
  }

  static inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    gmm::mult(gmm::transposed(A),B,X);
  }

  static inline void axpy(const real coef, const gene_vector & X, gene_vector & Y, int N){
    gmm::add(gmm::scaled(X,coef), Y);
  }

  static inline void axpby(real a, const gene_vector & X, real b, gene_vector & Y, int N){
    gmm::add(gmm::scaled(X,a), gmm::scaled(Y,b), Y);
  }

  static inline void copy_matrix(const gene_matrix & source, gene_matrix & cible, int N){
    gmm::copy(source,cible);
  }

  static inline void copy_vector(const gene_vector & source, gene_vector & cible, int N){
    gmm::copy(source,cible);
  }

  static inline void trisolve_lower(const gene_matrix & L, const gene_vector& B, gene_vector & X, int N){
    gmm::copy(B,X);
    gmm::lower_tri_solve(L, X, false);
  }

  static inline void partial_lu_decomp(const gene_matrix & X, gene_matrix & R, int N){
    gmm::copy(X,R);
    std::vector<int> ipvt(N);
    gmm::lu_factor(R, ipvt);
  }

  static inline void hessenberg(const gene_matrix & X, gene_matrix & R, int N){
    gmm::copy(X,R);
    gmm::Hessenberg_reduction(R,X,false);
  }

  static inline void tridiagonalization(const gene_matrix & X, gene_matrix & R, int N){
    gmm::copy(X,R);
    gmm::Householder_tridiagonalization(R,X,false);
  }

};

#endif
