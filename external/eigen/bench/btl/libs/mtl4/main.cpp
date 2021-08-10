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
#include "utilities.h"
#include "mtl4_interface.hh"
#include "bench.hh"
#include "basic_actions.hh"
#include "action_cholesky.hh"
// #include "action_lu_decomp.hh"

BTL_MAIN;

int main()
{

  bench<Action_axpy<mtl4_interface<REAL_TYPE> > >(MIN_AXPY,MAX_AXPY,NB_POINT);
  bench<Action_axpby<mtl4_interface<REAL_TYPE> > >(MIN_AXPY,MAX_AXPY,NB_POINT);

  bench<Action_matrix_vector_product<mtl4_interface<REAL_TYPE> > >(MIN_MV,MAX_MV,NB_POINT);
  bench<Action_atv_product<mtl4_interface<REAL_TYPE> > >(MIN_MV,MAX_MV,NB_POINT);
  bench<Action_matrix_matrix_product<mtl4_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
//   bench<Action_ata_product<mtl4_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
//   bench<Action_aat_product<mtl4_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);

  bench<Action_trisolve<mtl4_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
//   bench<Action_cholesky<mtl4_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
//   bench<Action_lu_decomp<mtl4_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);

  return 0;
}


