//=====================================================
// File   :  ublas_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  lun sep 30 14:23:27 CEST 2002
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
#ifndef UBLAS_INTERFACE_HH
#define UBLAS_INTERFACE_HH

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>

using namespace boost::numeric;

template <class real>
class ublas_interface{

public :

  typedef real real_type ;

  typedef std::vector<real> stl_vector;
  typedef std::vector<stl_vector> stl_matrix;

  typedef typename boost::numeric::ublas::matrix<real,boost::numeric::ublas::column_major> gene_matrix;
  typedef typename boost::numeric::ublas::vector<real> gene_vector;

  static inline std::string name( void ) { return "ublas"; }

  static void free_matrix(gene_matrix & A, int N) {}

  static void free_vector(gene_vector & B) {}

  static inline void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    A.resize(A_stl.size(),A_stl[0].size());
    for (int j=0; j<A_stl.size() ; j++)
      for (int i=0; i<A_stl[j].size() ; i++)
        A(i,j)=A_stl[j][i];
  }

  static inline void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    B.resize(B_stl.size());
    for (int i=0; i<B_stl.size() ; i++)
      B(i)=B_stl[i];
  }

  static inline void vector_to_stl(gene_vector & B, stl_vector & B_stl){
    for (int i=0; i<B_stl.size() ; i++)
      B_stl[i]=B(i);
  }

  static inline void matrix_to_stl(gene_matrix & A, stl_matrix & A_stl){
    int N=A_stl.size();
    for (int j=0;j<N;j++)
    {
      A_stl[j].resize(N);
      for (int i=0;i<N;i++)
        A_stl[j][i]=A(i,j);
    }
  }

  static inline void copy_vector(const gene_vector & source, gene_vector & cible, int N){
    for (int i=0;i<N;i++){
      cible(i) = source(i);
    }
  }

  static inline void copy_matrix(const gene_matrix & source, gene_matrix & cible, int N){
    for (int i=0;i<N;i++){
      for (int j=0;j<N;j++){
        cible(i,j) = source(i,j);
      }
    }
  }

  static inline void matrix_vector_product_slow(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    X =  prod(A,B);
  }

  static inline void matrix_matrix_product_slow(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N){
    X =  prod(A,B);
  }

  static inline void axpy_slow(const real coef, const gene_vector & X, gene_vector & Y, int N){
    Y+=coef*X;
  }

  // alias free assignements

  static inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    X.assign(prod(A,B));
  }

  static inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    X.assign(prod(trans(A),B));
  }

  static inline void matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N){
    X.assign(prod(A,B));
  }

  static inline void axpy(const real coef, const gene_vector & X, gene_vector & Y, int N){
    Y.plus_assign(coef*X);
  }

  static inline void axpby(real a, const gene_vector & X, real b, gene_vector & Y, int N){
    Y = a*X + b*Y;
  }

  static inline void ata_product(gene_matrix & A, gene_matrix & X, int N){
    // X =  prod(trans(A),A);
    X.assign(prod(trans(A),A));
  }

  static inline void aat_product(gene_matrix & A, gene_matrix & X, int N){
    // X =  prod(A,trans(A));
    X.assign(prod(A,trans(A)));
  }

  static inline void trisolve_lower(const gene_matrix & L, const gene_vector& B, gene_vector & X, int N){
    X = solve(L, B, ublas::lower_tag ());
  }

};

#endif
