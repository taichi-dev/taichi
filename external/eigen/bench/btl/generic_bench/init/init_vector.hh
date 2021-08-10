//=====================================================
// File   :  init_vector.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  lun sep 30 14:23:18 CEST 2002
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
#ifndef INIT_VECTOR_HH
#define INIT_VECTOR_HH

// The Vector class must satisfy the following part of STL vector concept :
//            resize() method
//            [] operator for setting element
//            value_type defined
template<double init_function(int), class Vector>
void init_vector(Vector & X, int size){

  X.resize(size);

  for (unsigned int i=0;i<X.size();i++){
    X[i]=typename Vector::value_type(init_function(i));
  }
}

#endif
