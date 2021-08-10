//=====================================================
// File   :  dump_file_x_y.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  lun sep 30 14:23:20 CEST 2002
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
#ifndef XY_FILE_HH
#define XY_FILE_HH
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

bool read_xy_file(const std::string & filename, std::vector<int> & tab_sizes,
                  std::vector<double> & tab_mflops, bool quiet = false)
{

  std::ifstream input_file (filename.c_str(),std::ios::in);

  if (!input_file){
    if (!quiet) {
      INFOS("!!! Error opening "<<filename);
    }
    return false;
  }

  int nb_point=0;
  int size=0;
  double mflops=0;

  while (input_file >> size >> mflops ){
    nb_point++;
    tab_sizes.push_back(size);
    tab_mflops.push_back(mflops);
  }
  SCRUTE(nb_point);

  input_file.close();
  return true;
}

// The Vector class must satisfy the following part of STL vector concept :
//            resize() method
//            [] operator for seting element
// the vector element must have the << operator define

using namespace std;

template<class Vector_A, class Vector_B>
void dump_xy_file(const Vector_A & X, const Vector_B & Y, const std::string & filename){
  
  ofstream outfile (filename.c_str(),ios::out) ;
  int size=X.size();
  
  for (int i=0;i<size;i++)
    outfile << X[i] << " " << Y[i] << endl;

  outfile.close();
} 

#endif
