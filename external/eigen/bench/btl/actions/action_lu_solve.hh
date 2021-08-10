//=====================================================
// File   :  action_lu_solve.hh
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
#ifndef ACTION_LU_SOLVE
#define ACTION_LU_SOLVE
#include "utilities.h"
#include "STL_interface.hh"
#include <string>
#include "init/init_function.hh"
#include "init/init_vector.hh"
#include "init/init_matrix.hh"

using namespace std;

template<class Interface>
class Action_lu_solve 
{

public :

  static inline std::string name( void )
  {
    return "lu_solve_"+Interface::name();
  }
  
  static double nb_op_base(int size){
    return 2.0*size*size*size/3.0;  // questionable but not really important
  }


  static double calculate( int nb_calc, int size ) {

    // STL matrix and vector initialization
    
    typename Interface::stl_matrix A_stl;
    typename Interface::stl_vector B_stl;
    typename Interface::stl_vector X_stl;

    init_matrix<pseudo_random>(A_stl,size);
    init_vector<pseudo_random>(B_stl,size);
    init_vector<null_function>(X_stl,size);

    // generic matrix and vector initialization

    typename Interface::gene_matrix A;
    typename Interface::gene_vector B;
    typename Interface::gene_vector X;

    typename Interface::gene_matrix LU; 

    Interface::matrix_from_stl(A,A_stl);
    Interface::vector_from_stl(B,B_stl);
    Interface::vector_from_stl(X,X_stl);
    Interface::matrix_from_stl(LU,A_stl);
  
    // local variable :

    typename Interface::Pivot_Vector pivot; // pivot vector
    Interface::new_Pivot_Vector(pivot,size);
    
    // timer utilities

    Portable_Timer chronos;

    // time measurement

    chronos.start();
    
    for (int ii=0;ii<nb_calc;ii++){

      // LU factorization
      Interface::copy_matrix(A,LU,size);
      Interface::LU_factor(LU,pivot,size);
      
      // LU solve

      Interface::LU_solve(LU,pivot,B,X,size);

    }

    // Time stop

    chronos.stop();

    double time=chronos.user_time();
  
    // check result :

    typename Interface::stl_vector B_new_stl(size);
    Interface::vector_to_stl(X,X_stl);

    STL_interface<typename Interface::real_type>::matrix_vector_product(A_stl,X_stl,B_new_stl,size); 
  
    typename Interface::real_type error=
      STL_interface<typename Interface::real_type>::norm_diff(B_stl,B_new_stl);
    
    if (error>1.e-5){
      INFOS("WRONG CALCULATION...residual=" << error);
      STL_interface<typename Interface::real_type>::display_vector(B_stl);
      STL_interface<typename Interface::real_type>::display_vector(B_new_stl);
      exit(0);
    }
    
    // deallocation and return time
    
    Interface::free_matrix(A,size);
    Interface::free_vector(B);
    Interface::free_vector(X);
    Interface::free_Pivot_Vector(pivot);

    return time;
  }

};
  

#endif



