//=====================================================
// File   :  smooth.cxx
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
#include <deque>
#include <string>
#include <iostream>
#include <fstream>
#include "bench_parameter.hh"
#include <set>

using namespace std;

void read_xy_file(const string & filename, vector<int> & tab_sizes, vector<double> & tab_mflops);
void write_xy_file(const string & filename, vector<int> & tab_sizes, vector<double> & tab_mflops);
void smooth_curve(const vector<double> & tab_mflops, vector<double> & smooth_tab_mflops,int window_half_width);
void centered_smooth_curve(const vector<double> & tab_mflops, vector<double> & smooth_tab_mflops,int window_half_width);

/////////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc , char *argv[] )
{

  // input data

  if (argc<3){
    INFOS("!!! Error ... usage : main filename window_half_width smooth_filename");
    exit(0);
  }
  INFOS(argc);

  int window_half_width=atoi(argv[2]);

  string filename=argv[1];
  string smooth_filename=argv[3];
  
  INFOS(filename);
  INFOS("window_half_width="<<window_half_width);

  vector<int> tab_sizes;
  vector<double> tab_mflops;

  read_xy_file(filename,tab_sizes,tab_mflops);

  // smoothing

  vector<double> smooth_tab_mflops;

  //smooth_curve(tab_mflops,smooth_tab_mflops,window_half_width);
  centered_smooth_curve(tab_mflops,smooth_tab_mflops,window_half_width);

  // output result

  write_xy_file(smooth_filename,tab_sizes,smooth_tab_mflops);
  

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class VECTOR>
double weighted_mean(const VECTOR & data)
{

  double mean=0.0;
  
  for (int i=0 ; i<data.size() ; i++){

    mean+=data[i];

  }

  return mean/double(data.size()) ;

}    




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void smooth_curve(const vector<double> & tab_mflops, vector<double> & smooth_tab_mflops,int window_half_width){
  
  int window_width=2*window_half_width+1;

  int size=tab_mflops.size();

  vector<double> sample(window_width);
  
  for (int i=0 ; i < size ; i++){
    
    for ( int j=0 ; j < window_width ; j++ ){
      
      int shifted_index=i+j-window_half_width;
      if (shifted_index<0) shifted_index=0;
      if (shifted_index>size-1) shifted_index=size-1;
      sample[j]=tab_mflops[shifted_index];
      
    }

    smooth_tab_mflops.push_back(weighted_mean(sample));

  }

}

void centered_smooth_curve(const vector<double> & tab_mflops, vector<double> & smooth_tab_mflops,int window_half_width){
  
  int max_window_width=2*window_half_width+1;

  int size=tab_mflops.size();

  
  for (int i=0 ; i < size ; i++){

    deque<double> sample;

    
    sample.push_back(tab_mflops[i]);

    for ( int j=1 ; j <= window_half_width ; j++ ){
      
      int before=i-j;
      int after=i+j;
      
      if ((before>=0)&&(after<size)) // inside of the vector
	{ 
	  sample.push_front(tab_mflops[before]);
	  sample.push_back(tab_mflops[after]);
	}
    }
    
    smooth_tab_mflops.push_back(weighted_mean(sample));
    
  }

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_xy_file(const string & filename, vector<int> & tab_sizes, vector<double> & tab_mflops){

  ofstream output_file (filename.c_str(),ios::out) ;
  
  for (int i=0 ; i < tab_sizes.size() ; i++)
    {
      output_file << tab_sizes[i] << " " <<  tab_mflops[i] << endl ;
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

