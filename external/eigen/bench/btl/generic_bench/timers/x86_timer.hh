//=====================================================
// File   :  x86_timer.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  mar d�c 3 18:59:35 CET 2002
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
#ifndef _X86_TIMER_HH
#define _X86_TIMER_HH

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/times.h>
//#include "system_time.h"
#define u32 unsigned int
#include <asm/msr.h>
#include "utilities.h"
#include <map>
#include <fstream>
#include <string>
#include <iostream>

// frequence de la becanne en Hz
//#define FREQUENCY 648000000
//#define FREQUENCY 1400000000
#define FREQUENCY 1695000000

using namespace std;


class X86_Timer {

public :

  X86_Timer( void ):_frequency(FREQUENCY),_nb_sample(0)
  {
    MESSAGE("X86_Timer Default Ctor");    
  }

  inline void start( void ){

    rdtsc(_click_start.n32[0],_click_start.n32[1]);

  }


  inline void stop( void ){

    rdtsc(_click_stop.n32[0],_click_stop.n32[1]);

  }
  

  inline double frequency( void ){
    return _frequency;
  }

  double get_elapsed_time_in_second( void ){

    return (_click_stop.n64-_click_start.n64)/double(FREQUENCY);


  }    

  unsigned long long  get_click( void ){
    
    return (_click_stop.n64-_click_start.n64);

  }    

  inline void find_frequency( void ){

    time_t initial, final;
    int dummy=2;

    initial = time(0);
    start();
    do {
      dummy+=2;
    }
    while(time(0)==initial);
    // On est au debut d'un cycle d'une seconde !!!
    initial = time(0);
    start();
    do {
      dummy+=2;
    }
    while(time(0)==initial);
    final=time(0);
    stop();
    //    INFOS("fine grained time : "<<  get_elapsed_time_in_second());
    //  INFOS("coarse grained time : "<<  final-initial);
    _frequency=_frequency*get_elapsed_time_in_second()/double(final-initial);
    ///  INFOS("CPU frequency : "<<  _frequency);        

  }

  void  add_get_click( void ){
       
    _nb_sample++;
    _counted_clicks[get_click()]++;
    fill_history_clicks();

  }    

  void dump_statistics(string filemane){
    
    ofstream outfile (filemane.c_str(),ios::out) ;

    std::map<unsigned long long , unsigned long long>::iterator itr;
    for(itr=_counted_clicks.begin() ; itr!=_counted_clicks.end()  ; itr++)
      {      
      outfile  << (*itr).first << "  " << (*itr).second << endl ;       
      }      
    
    outfile.close();

  }

  void dump_history(string filemane){
    
    ofstream outfile (filemane.c_str(),ios::out) ;



    for(int i=0 ; i<_history_mean_clicks.size() ; i++)
      {      
	outfile  << i << " " 
		 << _history_mean_clicks[i] << " " 
		 << _history_shortest_clicks[i] << " " 
		 << _history_most_occured_clicks[i] << endl ;
      }      
    
    outfile.close();

  }
     


  double get_mean_clicks( void ){
    
    std::map<unsigned long long,unsigned long long>::iterator itr;
    
    unsigned long long mean_clicks=0;

    for(itr=_counted_clicks.begin() ; itr!=_counted_clicks.end()  ; itr++)
      {      
	
	mean_clicks+=(*itr).second*(*itr).first;
      }      

    return mean_clicks/double(_nb_sample);

  }

  double get_shortest_clicks( void ){
    
    return double((*_counted_clicks.begin()).first);

  }

  void fill_history_clicks( void ){

    _history_mean_clicks.push_back(get_mean_clicks());
    _history_shortest_clicks.push_back(get_shortest_clicks());
    _history_most_occured_clicks.push_back(get_most_occured_clicks());

  }


  double get_most_occured_clicks( void ){

    unsigned long long moc=0;
    unsigned long long max_occurence=0;

    std::map<unsigned long long,unsigned long long>::iterator itr;

    for(itr=_counted_clicks.begin() ; itr!=_counted_clicks.end()  ; itr++)
      {      
	
	if (max_occurence<=(*itr).second){
	  max_occurence=(*itr).second;
	  moc=(*itr).first;
	}
      }      
    
    return double(moc);    

  }
  
  void clear( void )
  {
    _counted_clicks.clear();

    _history_mean_clicks.clear();
    _history_shortest_clicks.clear();
    _history_most_occured_clicks.clear();

    _nb_sample=0;
  }


    
private :
  
  union
  {
    unsigned long int n32[2] ;
    unsigned long long n64 ;
  } _click_start;

  union
  {
    unsigned long int n32[2] ;
    unsigned long long n64 ;
  } _click_stop;

  double _frequency ;

  map<unsigned long long,unsigned long long> _counted_clicks;

  vector<double> _history_mean_clicks;
  vector<double> _history_shortest_clicks;
  vector<double> _history_most_occured_clicks;

  unsigned long long _nb_sample;

  

};


#endif
