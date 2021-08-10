//=====================================================
// File   :  blas_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  lun sep 30 14:23:28 CEST 2002
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
#ifndef blas_PRODUIT_MATRICE_VECTEUR_HH
#define blas_PRODUIT_MATRICE_VECTEUR_HH

#include <c_interface_base.h>
#include <complex>
extern "C"
{
#include "blas.h"

  // Cholesky Factorization
//   void spotrf_(const char* uplo, const int* n, float *a, const int* ld, int* info);
//   void dpotrf_(const char* uplo, const int* n, double *a, const int* ld, int* info);
  void ssytrd_(char *uplo, const int *n, float *a, const int *lda, float *d, float *e, float *tau, float *work, int *lwork, int *info );
  void dsytrd_(char *uplo, const int *n, double *a, const int *lda, double *d, double *e, double *tau, double *work, int *lwork, int *info );
  void sgehrd_( const int *n, int *ilo, int *ihi, float *a, const int *lda, float *tau, float *work, int *lwork, int *info );
  void dgehrd_( const int *n, int *ilo, int *ihi, double *a, const int *lda, double *tau, double *work, int *lwork, int *info );

  // LU row pivoting
//   void dgetrf_( int *m, int *n, double *a, int *lda, int *ipiv, int *info );
//   void sgetrf_(const int* m, const int* n, float *a, const int* ld, int* ipivot, int* info);
  // LU full pivoting
  void sgetc2_(const int* n, float *a, const int *lda, int *ipiv, int *jpiv, int*info );
  void dgetc2_(const int* n, double *a, const int *lda, int *ipiv, int *jpiv, int*info );
#ifdef HAS_LAPACK
#endif
}

#define MAKE_STRING2(S) #S
#define MAKE_STRING(S) MAKE_STRING2(S)

#define CAT2(A,B) A##B
#define CAT(A,B) CAT2(A,B)


template<class real> class blas_interface;


static char notrans = 'N';
static char trans = 'T';
static char nonunit = 'N';
static char lower = 'L';
static char right = 'R';
static char left = 'L';
static int intone = 1;



#define SCALAR        float
#define SCALAR_PREFIX s
#include "blas_interface_impl.hh"
#undef SCALAR
#undef SCALAR_PREFIX


#define SCALAR        double
#define SCALAR_PREFIX d
#include "blas_interface_impl.hh"
#undef SCALAR
#undef SCALAR_PREFIX

#endif



