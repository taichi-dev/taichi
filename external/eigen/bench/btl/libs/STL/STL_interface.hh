//=====================================================
// File   :  STL_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  lun sep 30 14:23:24 CEST 2002
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
#ifndef STL_INTERFACE_HH
#define STL_INTERFACE_HH
#include <string>
#include <vector>
#include "utilities.h"

using namespace std;

template<class real>
class STL_interface{

public :

  typedef real real_type ;

  typedef std::vector<real>  stl_vector;
  typedef std::vector<stl_vector > stl_matrix;

  typedef stl_matrix gene_matrix;

  typedef stl_vector gene_vector;

  static inline std::string name( void )
  {
    return "STL";
  }

  static void free_matrix(gene_matrix & /*A*/, int /*N*/){}

  static void free_vector(gene_vector & /*B*/){}

  static inline void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    A = A_stl;
  }

  static inline void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    B = B_stl;
  }

  static inline void vector_to_stl(gene_vector & B, stl_vector & B_stl){
    B_stl = B ;
  }


  static inline void matrix_to_stl(gene_matrix & A, stl_matrix & A_stl){
    A_stl = A ;
  }

  static inline void copy_vector(const gene_vector & source, gene_vector & cible, int N){
    for (int i=0;i<N;i++){
      cible[i]=source[i];
    }
  }


  static inline void copy_matrix(const gene_matrix & source, gene_matrix & cible, int N){
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
        cible[i][j]=source[i][j];
  }

//   static inline void ata_product(const gene_matrix & A, gene_matrix & X, int N)
//   {
//     real somme;
//     for (int j=0;j<N;j++){
//       for (int i=0;i<N;i++){
//         somme=0.0;
//         for (int k=0;k<N;k++)
//           somme += A[i][k]*A[j][k];
//         X[j][i]=somme;
//       }
//     }
//   }

  static inline void aat_product(const gene_matrix & A, gene_matrix & X, int N)
  {
    real somme;
    for (int j=0;j<N;j++){
      for (int i=0;i<N;i++){
        somme=0.0;
        if(i>=j)
        {
          for (int k=0;k<N;k++){
            somme+=A[k][i]*A[k][j];
          }
          X[j][i]=somme;
        }
      }
    }
  }


  static inline void matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int N)
  {
    real somme;
    for (int j=0;j<N;j++){
      for (int i=0;i<N;i++){
        somme=0.0;
        for (int k=0;k<N;k++)
          somme+=A[k][i]*B[j][k];
        X[j][i]=somme;
      }
    }
  }

  static inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N)
  {
    real somme;
    for (int i=0;i<N;i++){
      somme=0.0;
      for (int j=0;j<N;j++)
        somme+=A[j][i]*B[j];
      X[i]=somme;
    }
  }

  static inline void symv(gene_matrix & A, gene_vector & B, gene_vector & X, int N)
  {
    for (int j=0; j<N; ++j)
      X[j] = 0;
    for (int j=0; j<N; ++j)
    {
      real t1 = B[j];
      real t2 = 0;
      X[j] += t1 * A[j][j];
      for (int i=j+1; i<N; ++i) {
        X[i] += t1 * A[j][i];
        t2 += A[j][i] * B[i];
      }
      X[j] += t2;
    }
  }
  
  static inline void syr2(gene_matrix & A, gene_vector & B, gene_vector & X, int N)
  {
    for (int j=0; j<N; ++j)
    {
      for (int i=j; i<N; ++i)
        A[j][i] += B[i]*X[j] + B[j]*X[i];
    }
  }

  static inline void ger(gene_matrix & A, gene_vector & X, gene_vector & Y, int N)
  {
    for (int j=0; j<N; ++j)
    {
      for (int i=j; i<N; ++i)
        A[j][i] += X[i]*Y[j];
    }
  }

  static inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N)
  {
    real somme;
    for (int i=0;i<N;i++){
      somme = 0.0;
      for (int j=0;j<N;j++)
        somme += A[i][j]*B[j];
      X[i] = somme;
    }
  }

  static inline void axpy(real coef, const gene_vector & X, gene_vector & Y, int N){
    for (int i=0;i<N;i++)
      Y[i]+=coef*X[i];
  }

  static inline void axpby(real a, const gene_vector & X, real b, gene_vector & Y, int N){
    for (int i=0;i<N;i++)
      Y[i] = a*X[i] + b*Y[i];
  }

  static inline void trisolve_lower(const gene_matrix & L, const gene_vector & B, gene_vector & X, int N){
    copy_vector(B,X,N);
    for(int i=0; i<N; ++i)
    {
      X[i] /= L[i][i];
      real tmp = X[i];
      for (int j=i+1; j<N; ++j)
        X[j] -= tmp * L[i][j];
    }
  }

  static inline real norm_diff(const stl_vector & A, const stl_vector & B)
  {
    int N=A.size();
    real somme=0.0;
    real somme2=0.0;

    for (int i=0;i<N;i++){
      real diff=A[i]-B[i];
      somme+=diff*diff;
      somme2+=A[i]*A[i];
    }
    return somme/somme2;
  }

  static inline real norm_diff(const stl_matrix & A, const stl_matrix & B)
  {
    int N=A[0].size();
    real somme=0.0;
    real somme2=0.0;

    for (int i=0;i<N;i++){
      for (int j=0;j<N;j++){
        real diff=A[i][j] - B[i][j];
        somme += diff*diff;
        somme2 += A[i][j]*A[i][j];
      }
    }

    return somme/somme2;
  }

  static inline void display_vector(const stl_vector & A)
  {
    int N=A.size();
    for (int i=0;i<N;i++){
      INFOS("A["<<i<<"]="<<A[i]<<endl);
    }
  }

};

#endif
