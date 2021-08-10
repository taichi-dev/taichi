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
#include "eigen2_interface.hh"
#include "bench.hh"
#include "action_trisolve.hh"
#include "action_trisolve_matrix.hh"
#include "action_cholesky.hh"
#include "action_hessenberg.hh"
#include "action_lu_decomp.hh"
// #include "action_partial_lu.hh"

BTL_MAIN;

int main()
{
  bench<Action_trisolve<eigen2_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
  bench<Action_trisolve_matrix<eigen2_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
  bench<Action_cholesky<eigen2_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
  bench<Action_lu_decomp<eigen2_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
//   bench<Action_partial_lu<eigen2_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);

  bench<Action_hessenberg<eigen2_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);
  bench<Action_tridiagonalization<eigen2_interface<REAL_TYPE> > >(MIN_MM,MAX_MM,NB_POINT);

  return 0;
}


