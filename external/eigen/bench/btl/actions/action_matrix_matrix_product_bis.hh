//=====================================================
// File   :  action_matrix_matrix_product_bis.hh
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
#ifndef ACTION_MATRIX_MATRIX_PRODUCT_BIS
#define ACTION_MATRIX_MATRIX_PRODUCT_BIS
#include "utilities.h"
#include "STL_interface.hh"
#include "STL_timer.hh"
#include <string>
#include "init_function.hh"
#include "init_vector.hh"
#include "init_matrix.hh"

using namespace std;

template<class Interface>
class Action_matrix_matrix_product_bis {

public :

  static inline std::string name( void )
  {
    return "matrix_matrix_"+Interface::name();
  }

  static double nb_op_base(int size){
    return 2.0*size*size*size;
  }

  static double calculate( int nb_calc, int size ) {

    // STL matrix and vector initialization

    typename Interface::stl_matrix A_stl;
    typename Interface::stl_matrix B_stl;
    typename Interface::stl_matrix X_stl;

    init_matrix<pseudo_random>(A_stl,size);
    init_matrix<pseudo_random>(B_stl,size);
    init_matrix<null_function>(X_stl,size);

    // generic matrix and vector initialization

    typename Interface::gene_matrix A_ref;
    typename Interface::gene_matrix B_ref;
    typename Interface::gene_matrix X_ref;

    typename Interface::gene_matrix A;
    typename Interface::gene_matrix B;
    typename Interface::gene_matrix X;


    Interface::matrix_from_stl(A_ref,A_stl);
    Interface::matrix_from_stl(B_ref,B_stl);
    Interface::matrix_from_stl(X_ref,X_stl);

    Interface::matrix_from_stl(A,A_stl);
    Interface::matrix_from_stl(B,B_stl);
    Interface::matrix_from_stl(X,X_stl);


    // STL_timer utilities

    STL_timer chronos;

    // Baseline evaluation

    chronos.start_baseline(nb_calc);

    do {

      Interface::copy_matrix(A_ref,A,size);
      Interface::copy_matrix(B_ref,B,size);
      Interface::copy_matrix(X_ref,X,size);


      //      Interface::matrix_matrix_product(A,B,X,size); This line must be commented !!!!
    }
    while(chronos.check());

    chronos.report(true);

    // Time measurement

    chronos.start(nb_calc);

    do {

      Interface::copy_matrix(A_ref,A,size);
      Interface::copy_matrix(B_ref,B,size);
      Interface::copy_matrix(X_ref,X,size);

      Interface::matrix_matrix_product(A,B,X,size); // here it is not commented !!!!
    }
    while(chronos.check());

    chronos.report(true);

    double time=chronos.calculated_time/2000.0;

    // calculation check

    typename Interface::stl_matrix resu_stl(size);

    Interface::matrix_to_stl(X,resu_stl);

    STL_interface<typename Interface::real_type>::matrix_matrix_product(A_stl,B_stl,X_stl,size);

    typename Interface::real_type error=
      STL_interface<typename Interface::real_type>::norm_diff(X_stl,resu_stl);

    if (error>1.e-6){
      INFOS("WRONG CALCULATION...residual=" << error);
      exit(1);
    }

    // deallocation and return time

    Interface::free_matrix(A,size);
    Interface::free_matrix(B,size);
    Interface::free_matrix(X,size);

    Interface::free_matrix(A_ref,size);
    Interface::free_matrix(B_ref,size);
    Interface::free_matrix(X_ref,size);

    return time;
  }

};


#endif



