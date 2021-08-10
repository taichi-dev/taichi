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
#include "eigen3_interface.hh"
#include "static/bench_static.hh"
#include "action_matrix_vector_product.hh"
#include "action_matrix_matrix_product.hh"
#include "action_axpy.hh"
#include "action_lu_solve.hh"
#include "action_ata_product.hh"
#include "action_aat_product.hh"
#include "action_atv_product.hh"
#include "action_cholesky.hh"
#include "action_trisolve.hh"

BTL_MAIN;

int main()
{

  bench_static<Action_axpy,eigen2_interface>();
  bench_static<Action_matrix_matrix_product,eigen2_interface>();
  bench_static<Action_matrix_vector_product,eigen2_interface>();
  bench_static<Action_atv_product,eigen2_interface>();
  bench_static<Action_cholesky,eigen2_interface>();
  bench_static<Action_trisolve,eigen2_interface>();

  return 0;
}


