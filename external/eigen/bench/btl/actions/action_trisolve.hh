//=====================================================
// File   :  action_trisolve.hh
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
#ifndef ACTION_TRISOLVE
#define ACTION_TRISOLVE
#include "utilities.h"
#include "STL_interface.hh"
#include <string>
#include "init/init_function.hh"
#include "init/init_vector.hh"
#include "init/init_matrix.hh"

using namespace std;

template<class Interface>
class Action_trisolve {

public :

  // Ctor

  Action_trisolve( int size ):_size(size)
  {
    MESSAGE("Action_trisolve Ctor");

    // STL vector initialization
    init_matrix<pseudo_random>(L_stl,_size);
    init_vector<pseudo_random>(B_stl,_size);
    init_vector<null_function>(X_stl,_size);
    for (int j=0; j<_size; ++j)
    {
      for (int i=0; i<j; ++i)
        L_stl[j][i] = 0;
      L_stl[j][j] += 3;
    }

    init_vector<null_function>(resu_stl,_size);

    // generic matrix and vector initialization
    Interface::matrix_from_stl(L,L_stl);
    Interface::vector_from_stl(X,X_stl);
    Interface::vector_from_stl(B,B_stl);

    _cost = 0;
    for (int j=0; j<_size; ++j)
    {
      _cost += 2*j + 1;
    }
  }

  // invalidate copy ctor

  Action_trisolve( const  Action_trisolve & )
  {
    INFOS("illegal call to Action_trisolve Copy Ctor");
    exit(1);
  }

  // Dtor

  ~Action_trisolve( void ){

    MESSAGE("Action_trisolve Dtor");

    // deallocation
    Interface::free_matrix(L,_size);
    Interface::free_vector(B);
    Interface::free_vector(X);
  }

  // action name

  static inline std::string name( void )
  {
    return "trisolve_vector_"+Interface::name();
  }

  double nb_op_base( void ){
    return _cost;
  }

  inline void initialize( void ){
    //Interface::copy_vector(X_ref,X,_size);
  }

  inline void calculate( void ) {
      Interface::trisolve_lower(L,B,X,_size);
  }

  void check_result(){
    if (_size>128) return;
    // calculation check
    Interface::vector_to_stl(X,resu_stl);

    STL_interface<typename Interface::real_type>::trisolve_lower(L_stl,B_stl,X_stl,_size);

    typename Interface::real_type error=
      STL_interface<typename Interface::real_type>::norm_diff(X_stl,resu_stl);

    if (error>1.e-4){
      INFOS("WRONG CALCULATION...residual=" << error);
      exit(2);
    } //else INFOS("CALCULATION OK...residual=" << error);

  }

private :

  typename Interface::stl_matrix L_stl;
  typename Interface::stl_vector X_stl;
  typename Interface::stl_vector B_stl;
  typename Interface::stl_vector resu_stl;

  typename Interface::gene_matrix L;
  typename Interface::gene_vector X;
  typename Interface::gene_vector B;

  int _size;
  double _cost;
};

#endif
