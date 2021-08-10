//=====================================================
// File   :  regularize.cxx
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  lun sep 30 14:23:15 CEST 2002
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
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "bench_parameter.hh"
#include <set>

using namespace std;

void read_xy_file(const string & filename, vector<int> & tab_sizes, vector<double> & tab_mflops);
void regularize_curve(const string & filename,
		      const vector<double> & tab_mflops, 
		      const vector<int> & tab_sizes, 
		      int start_cut_size, int stop_cut_size);
/////////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc , char *argv[] )
{

  // input data

  if (argc<4){
    INFOS("!!! Error ... usage : main filename start_cut_size stop_cut_size regularize_filename");
    exit(0);
  }
  INFOS(argc);

  int start_cut_size=atoi(argv[2]);
  int stop_cut_size=atoi(argv[3]);

  string filename=argv[1];
  string regularize_filename=argv[4];
  
  INFOS(filename);
  INFOS("start_cut_size="<<start_cut_size);

  vector<int> tab_sizes;
  vector<double> tab_mflops;

  read_xy_file(filename,tab_sizes,tab_mflops);

  // regularizeing

  regularize_curve(regularize_filename,tab_mflops,tab_sizes,start_cut_size,stop_cut_size);
  

}

//////////////////////////////////////////////////////////////////////////////////////

void regularize_curve(const string & filename,
		      const vector<double> & tab_mflops, 
		      const vector<int> & tab_sizes, 
		      int start_cut_size, int stop_cut_size)
{
  int size=tab_mflops.size();
  ofstream output_file (filename.c_str(),ios::out) ;

  int i=0;

  while(tab_sizes[i]<start_cut_size){
    
    output_file << tab_sizes[i] << " " <<  tab_mflops[i] << endl ;
    i++;

  }
    
  output_file << endl ;

  while(tab_sizes[i]<stop_cut_size){
    
    i++;

  }

  while(i<size){
    
    output_file << tab_sizes[i] << " " <<  tab_mflops[i] << endl ;
    i++;

  }

  output_file.close();

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void read_xy_file(const string & filename, vector<int> & tab_sizes, vector<double> & tab_mflops){

  ifstream input_file (filename.c_str(),ios::in) ;

  if (!input_file){
    INFOS("!!! Error opening "<<filename);
    exit(0);
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
}

