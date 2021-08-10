//=====================================================
// File   :  mean.cxx
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
#include "utils/xy_file.hh"
#include <set>

using namespace std;

double mean_calc(const vector<int> & tab_sizes, const vector<double> & tab_mflops, const int size_min, const int size_max);

class Lib_Mean{

public:
  Lib_Mean( void ):_lib_name(),_mean_in_cache(),_mean_out_of_cache(){
    MESSAGE("Lib_mean Default Ctor");
    MESSAGE("!!! should not be used");
    exit(0);
  }
  Lib_Mean(const string & name, const double & mic, const double & moc):_lib_name(name),_mean_in_cache(mic),_mean_out_of_cache(moc){
    MESSAGE("Lib_mean Ctor");
  }
  Lib_Mean(const Lib_Mean & lm):_lib_name(lm._lib_name),_mean_in_cache(lm._mean_in_cache),_mean_out_of_cache(lm._mean_out_of_cache){
    MESSAGE("Lib_mean Copy Ctor");
  }
  ~Lib_Mean( void ){
    MESSAGE("Lib_mean Dtor");
  }
    
  double _mean_in_cache;
  double _mean_out_of_cache;
  string _lib_name;

  bool operator < ( const Lib_Mean &right) const 
  {
    //return ( this->_mean_out_of_cache > right._mean_out_of_cache) ;
    return ( this->_mean_in_cache > right._mean_in_cache) ;
  }

}; 


int main( int argc , char *argv[] )
{

  if (argc<6){
    INFOS("!!! Error ... usage : main what mic Mic moc Moc filename1 finename2...");
    exit(0);
  }
  INFOS(argc);

  int min_in_cache=atoi(argv[2]);
  int max_in_cache=atoi(argv[3]);
  int min_out_of_cache=atoi(argv[4]);
  int max_out_of_cache=atoi(argv[5]);


  multiset<Lib_Mean> s_lib_mean ;

  for (int i=6;i<argc;i++){
    
    string filename=argv[i];
    
    INFOS(filename);

    double mic=0;
    double moc=0;

    {
      
      vector<int> tab_sizes;
      vector<double> tab_mflops;

      read_xy_file(filename,tab_sizes,tab_mflops);

      mic=mean_calc(tab_sizes,tab_mflops,min_in_cache,max_in_cache);
      moc=mean_calc(tab_sizes,tab_mflops,min_out_of_cache,max_out_of_cache);

      Lib_Mean cur_lib_mean(filename,mic,moc);
      
      s_lib_mean.insert(cur_lib_mean);	

    }   
           
  }


  cout << "<TABLE BORDER CELLPADDING=2>" << endl ;
  cout << "  <TR>" << endl ;
  cout << "    <TH ALIGN=CENTER> " << argv[1] << " </TH>" << endl ;
  cout << "    <TH ALIGN=CENTER> <a href=""#mean_marker""> in cache <BR> mean perf <BR> Mflops </a></TH>" << endl ;
  cout << "    <TH ALIGN=CENTER> in cache <BR> % best </TH>" << endl ;
  cout << "    <TH ALIGN=CENTER> <a href=""#mean_marker""> out of cache <BR> mean perf <BR> Mflops </a></TH>" << endl ;
  cout << "    <TH ALIGN=CENTER> out of cache <BR> % best </TH>" << endl ;
  cout << "    <TH ALIGN=CENTER> details </TH>" << endl ;
  cout << "    <TH ALIGN=CENTER> comments </TH>" << endl ;
  cout << "  </TR>" << endl ;

  multiset<Lib_Mean>::iterator is = s_lib_mean.begin();
  Lib_Mean best(*is);  
  

  for (is=s_lib_mean.begin(); is!=s_lib_mean.end() ; is++){

    cout << "  <TR>" << endl ;
    cout << "     <TD> " << is->_lib_name << " </TD>" << endl ;
    cout << "     <TD> " << is->_mean_in_cache << " </TD>" << endl ;
    cout << "     <TD> " << 100*(is->_mean_in_cache/best._mean_in_cache) << " </TD>" << endl ;
    cout << "     <TD> " << is->_mean_out_of_cache << " </TD>" << endl ;
    cout << "     <TD> " << 100*(is->_mean_out_of_cache/best._mean_out_of_cache) << " </TD>" << endl ;
    cout << "     <TD> " << 
      "<a href=\"#"<<is->_lib_name<<"_"<<argv[1]<<"\">snippet</a>/" 
      "<a href=\"#"<<is->_lib_name<<"_flags\">flags</a>  </TD>" << endl ;
    cout << "     <TD> " << 
      "<a href=\"#"<<is->_lib_name<<"_comments\">click here</a>  </TD>" << endl ;
    cout << "  </TR>" << endl ;
  
  }

  cout << "</TABLE>" << endl ;

  ofstream output_file ("../order_lib",ios::out) ;
  
  for (is=s_lib_mean.begin(); is!=s_lib_mean.end() ; is++){
    output_file << is->_lib_name << endl ;
  }

  output_file.close();

}

double mean_calc(const vector<int> & tab_sizes, const vector<double> & tab_mflops, const int size_min, const int size_max){
  
  int size=tab_sizes.size();
  int nb_sample=0;
  double mean=0.0;

  for (int i=0;i<size;i++){
    
    
    if ((tab_sizes[i]>=size_min)&&(tab_sizes[i]<=size_max)){
      
      nb_sample++;
      mean+=tab_mflops[i];

    }

    
  }

  if (nb_sample==0){
    INFOS("no data for mean calculation");
    return 0.0;
  }

  return mean/nb_sample;
}

  


