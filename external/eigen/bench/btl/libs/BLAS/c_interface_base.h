
#ifndef BTL_C_INTERFACE_BASE_H
#define BTL_C_INTERFACE_BASE_H

#include "utilities.h"
#include <vector>

template<class real> class c_interface_base
{

public:

  typedef real                      real_type;
  typedef std::vector<real>         stl_vector;
  typedef std::vector<stl_vector >  stl_matrix;

  typedef real* gene_matrix;
  typedef real* gene_vector;

  static void free_matrix(gene_matrix & A, int /*N*/){
    delete[] A;
  }

  static void free_vector(gene_vector & B){
    delete[] B;
  }

  static inline void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    int N = A_stl.size();
    A = new real[N*N];
    for (int j=0;j<N;j++)
      for (int i=0;i<N;i++)
        A[i+N*j] = A_stl[j][i];
  }

  static inline void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    int N = B_stl.size();
    B = new real[N];
    for (int i=0;i<N;i++)
      B[i] = B_stl[i];
  }

  static inline void vector_to_stl(gene_vector & B, stl_vector & B_stl){
    int N = B_stl.size();
    for (int i=0;i<N;i++)
      B_stl[i] = B[i];
  }

  static inline void matrix_to_stl(gene_matrix & A, stl_matrix & A_stl){
    int N = A_stl.size();
    for (int j=0;j<N;j++){
      A_stl[j].resize(N);
      for (int i=0;i<N;i++)
        A_stl[j][i] = A[i+N*j];
    }
  }

  static inline void copy_vector(const gene_vector & source, gene_vector & cible, int N){
    for (int i=0;i<N;i++)
      cible[i]=source[i];
  }

  static inline void copy_matrix(const gene_matrix & source, gene_matrix & cible, int N){
    for (int j=0;j<N;j++){
      for (int i=0;i<N;i++){
        cible[i+N*j] = source[i+N*j];
      }
    }
  }

};

#endif
