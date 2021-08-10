//=====================================================
// File   :  tvmet_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  lun sep 30 14:23:30 CEST 2002
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
#ifndef TVMET_INTERFACE_HH
#define TVMET_INTERFACE_HH

#include <tvmet/tvmet.h>
#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

#include <vector>

using namespace tvmet;

template<class real, int SIZE>
class tvmet_interface{

public :

  typedef real real_type ;

  typedef std::vector<real>  stl_vector;
  typedef std::vector<stl_vector > stl_matrix;

  typedef Vector<real,SIZE> gene_vector;
  typedef Matrix<real,SIZE,SIZE> gene_matrix;

  static inline std::string name() { return "tiny_tvmet"; }

  static void free_matrix(gene_matrix & A, int N){}

  static void free_vector(gene_vector & B){}

  static inline void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    for (int j=0; j<A_stl.size() ; j++)
      for (int i=0; i<A_stl[j].size() ; i++)
        A(i,j) = A_stl[j][i];
  }

  static inline void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    for (int i=0; i<B_stl.size() ; i++)
      B[i]=B_stl[i];
  }

  static inline void vector_to_stl(gene_vector & B, stl_vector & B_stl){
    for (int i=0; i<B_stl.size() ; i++){
      B_stl[i]=B[i];
    }
  }

  static inline void matrix_to_stl(gene_matrix & A, stl_matrix & A_stl){
    int N = A_stl.size();
    for (int j=0;j<N;j++){
      A_stl[j].resize(N);
      for (int i=0;i<N;i++)
        A_stl[j][i] = A(i,j);
    }
  }


  static inline void copy_matrix(const gene_matrix & source, gene_matrix & cible, int N){
    cible = source;
  }

  static inline void copy_vector(const gene_vector & source, gene_vector & cible, int N){
    cible = source;
  }

  static inline void matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int N){
    X = prod(A,B);
  }

  static inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    X = prod(A,B);
  }

  static inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    X = prod(trans(A),B);
  }

  static inline void axpy(const real coef, const gene_vector & X, gene_vector & Y, int N){
    Y+=coef*X;
  }

};


#endif
