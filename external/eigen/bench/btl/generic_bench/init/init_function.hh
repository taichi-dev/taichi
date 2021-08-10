//=====================================================
// File   :  init_function.hh
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
#ifndef INIT_FUNCTION_HH
#define INIT_FUNCTION_HH

double simple_function(int index)
{
  return index;
}

double simple_function(int index_i, int index_j)
{
  return index_i+index_j;
}

double pseudo_random(int /*index*/)
{
  return std::rand()/double(RAND_MAX);
}

double pseudo_random(int /*index_i*/, int /*index_j*/)
{
  return std::rand()/double(RAND_MAX);
}


double null_function(int /*index*/)
{
  return 0.0;
}

double null_function(int /*index_i*/, int /*index_j*/)
{
  return 0.0;
}

#endif
