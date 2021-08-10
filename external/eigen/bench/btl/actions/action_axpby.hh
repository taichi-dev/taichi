//=====================================================
// File   :  action_axpby.hh
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
#ifndef ACTION_AXPBY
#define ACTION_AXPBY
#include "utilities.h"
#include "STL_interface.hh"
#include <string>
#include "init/init_function.hh"
#include "init/init_vector.hh"
#include "init/init_matrix.hh"

using namespace std;

template<class Interface>
class Action_axpby {

public :

  // Ctor
  Action_axpby( int size ):_alpha(0.5),_beta(0.95),_size(size)
  {
    MESSAGE("Action_axpby Ctor");

    // STL vector initialization
    init_vector<pseudo_random>(X_stl,_size);
    init_vector<pseudo_random>(Y_stl,_size);
    init_vector<null_function>(resu_stl,_size);

    // generic matrix and vector initialization
    Interface::vector_from_stl(X_ref,X_stl);
    Interface::vector_from_stl(Y_ref,Y_stl);

    Interface::vector_from_stl(X,X_stl);
    Interface::vector_from_stl(Y,Y_stl);
  }

  // invalidate copy ctor
  Action_axpby( const  Action_axpby & )
  {
    INFOS("illegal call to Action_axpby Copy Ctor");
    exit(1);
  }

  // Dtor
  ~Action_axpby( void ){
    MESSAGE("Action_axpby Dtor");

    // deallocation
    Interface::free_vector(X_ref);
    Interface::free_vector(Y_ref);

    Interface::free_vector(X);
    Interface::free_vector(Y);
  }

  // action name
  static inline std::string name( void )
  {
    return "axpby_"+Interface::name();
  }

  double nb_op_base( void ){
    return 3.0*_size;
  }

  inline void initialize( void ){
    Interface::copy_vector(X_ref,X,_size);
    Interface::copy_vector(Y_ref,Y,_size);
  }

  inline void calculate( void ) {
    BTL_ASM_COMMENT("mybegin axpby");
    Interface::axpby(_alpha,X,_beta,Y,_size);
    BTL_ASM_COMMENT("myend axpby");
  }

  void check_result( void ){
    if (_size>128) return;
    // calculation check
    Interface::vector_to_stl(Y,resu_stl);

    STL_interface<typename Interface::real_type>::axpby(_alpha,X_stl,_beta,Y_stl,_size);

    typename Interface::real_type error=
      STL_interface<typename Interface::real_type>::norm_diff(Y_stl,resu_stl);

    if (error>1.e-6){
      INFOS("WRONG CALCULATION...residual=" << error);
      exit(2);
    }
  }

private :

  typename Interface::stl_vector X_stl;
  typename Interface::stl_vector Y_stl;
  typename Interface::stl_vector resu_stl;

  typename Interface::gene_vector X_ref;
  typename Interface::gene_vector Y_ref;

  typename Interface::gene_vector X;
  typename Interface::gene_vector Y;

  typename Interface::real_type _alpha;
  typename Interface::real_type _beta;

  int _size;
};

#endif
