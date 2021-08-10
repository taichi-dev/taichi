//=====================================================
// File   :  main.cpp
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
#include "utilities.h"
#include "blas_interface.hh"
#include "bench.hh"
#include "basic_actions.hh"

#include "action_cholesky.hh"
#include "action_lu_decomp.hh"
#include "action_partial_lu.hh"
#include "action_trisolve_matrix.hh"

#ifdef HAS_LAPACK
#include "action_hessenberg.hh"
#endif

BTL_MAIN;

int main()
{

  bench<Action_axpy<blas_interface<REAL_TYPE> > >(MIN_AXPY,MAX_AXPY,NB_POINT);
  bench<Action_axpby<blas_interface<REAL_TYPE> > >(MIN_AXPY,MAX_AXPY,NB_POINT);

  bench<Action_matrix_vector_product<blas_interface<REAL_TYPE> > >(MIN_MV,MAX_MV,NB_POINT);
  bench<Action_atv_product<blas_interface<REAL_TYPE> > >(MIN_MV,MAX_MV,NB_POINT);
  bench<Action_symv<blas_interface<REAL_TYPE> > >(MIN_MV,MAX_MV,NB_POINT);
  bench<Action_syr2<blas_interface<REAL_TYPE> > >(MIN_MV,MAX_MV,NB_POINT);

  bench<Action_ger<blas_interface<REAL_TYPE> > >(MIN_MV,MAX_MV,NB_POINT);
  bench<Action_rot<blas_interface<REAL_TYPE> > >(MIN_AXPY,MAX_AXPY,NB_POINT);

  bench<Action_matrix_matrix_product<blas_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
//   bench<Action_ata_product<blas_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
  bench<Action_aat_product<blas_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);

  bench<Action_trisolve<blas_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
  bench<Action_trisolve_matrix<blas_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);

  bench<Action_trmm<blas_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);

  bench<Action_cholesky<blas_interface<REAL_TYPE> > >(MIN_LU,MAX_LU,NB_POINT);
  bench<Action_partial_lu<blas_interface<REAL_TYPE> > >(MIN_LU,MAX_LU,NB_POINT);

  #ifdef HAS_LAPACK
//   bench<Action_lu_decomp<blas_interface<REAL_TYPE> > >(MIN_LU,MAX_LU,NB_POINT);
  bench<Action_hessenberg<blas_interface<REAL_TYPE> > >(MIN_LU,MAX_LU,NB_POINT);
  bench<Action_tridiagonalization<blas_interface<REAL_TYPE> > >(MIN_LU,MAX_LU,NB_POINT);
  #endif

  //bench<Action_lu_solve<blas_LU_solve_interface<REAL_TYPE> > >(MIN_LU,MAX_LU,NB_POINT);

  return 0;
}


