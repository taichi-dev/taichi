//=====================================================
// File   :  init_matrix.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  lun sep 30 14:23:19 CEST 2002
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
#ifndef INIT_MATRIX_HH
#define INIT_MATRIX_HH

// The Vector class must satisfy the following part of STL vector concept :
//            resize() method
//            [] operator for setting element
//            value_type defined
template<double init_function(int,int), class Vector>
BTL_DONT_INLINE void init_row(Vector & X, int size, int row){

  X.resize(size);

  for (unsigned int j=0;j<X.size();j++){
    X[j]=typename Vector::value_type(init_function(row,j));
  }
}


// Matrix is a Vector of Vector
// The Matrix class must satisfy the following part of STL vector concept :
//            resize() method
//            [] operator for setting rows
template<double init_function(int,int),class Vector>
BTL_DONT_INLINE void init_matrix(Vector &  A, int size){
  A.resize(size);
  for (unsigned int row=0; row<A.size() ; row++){
    init_row<init_function>(A[row],size,row);
  }
}

template<double init_function(int,int),class Matrix>
BTL_DONT_INLINE void init_matrix_symm(Matrix&  A, int size){
  A.resize(size);
  for (unsigned int row=0; row<A.size() ; row++)
    A[row].resize(size);
  for (unsigned int row=0; row<A.size() ; row++){
    A[row][row] = init_function(row,row);
    for (unsigned int col=0; col<row ; col++){
      double x = init_function(row,col);
      A[row][col] = A[col][row] = x;
    }
  }
}

#endif
