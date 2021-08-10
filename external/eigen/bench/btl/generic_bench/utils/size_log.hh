//=====================================================
// File   :  size_log.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  lun sep 30 14:23:17 CEST 2002
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
#ifndef SIZE_LOG
#define SIZE_LOG

#include "math.h"
// The Vector class must satisfy the following part of STL vector concept :
//            resize() method
//            [] operator for seting element
// the vector element are int compatible.
template<class Vector>
void size_log(const int nb_point, const int size_min, const int size_max, Vector & X)
{
  X.resize(nb_point);

  float ls_min=log(float(size_min));
  float ls_max=log(float(size_max));

  float ls=0.0;

  float delta_ls=(ls_max-ls_min)/(float(nb_point-1));

  int size=0;

  for (int i=0;i<nb_point;i++){

    ls = ls_min + float(i)*delta_ls ;
    
    size=int(exp(ls)); 

    X[i]=size;
  }

}


#endif
