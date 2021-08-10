//=====================================================
// File   :  size_lin_log.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  mar déc 3 18:59:37 CET 2002
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
#ifndef SIZE_LIN_LOG
#define SIZE_LIN_LOG

#include "size_log.hh"

template<class Vector>
void size_lin_log(const int nb_point, const int /*size_min*/, const int size_max, Vector & X)
{
  int ten=10;
  int nine=9;

  X.resize(nb_point);

  if (nb_point>ten){

    for (int i=0;i<nine;i++){
      
      X[i]=i+1;

    }

    Vector log_size;
    size_log(nb_point-nine,ten,size_max,log_size);

    for (int i=0;i<nb_point-nine;i++){
      
      X[i+nine]=log_size[i];

    }
  }
  else{

    for (int i=0;i<nb_point;i++){
      
      X[i]=i+1;

    }
  }

 //  for (int i=0;i<nb_point;i++){
    
//        INFOS("computed sizes : X["<<i<<"]="<<X[i]);
    
//   }

}
  
#endif
    


