//=====================================================
// File   :  STL_perf_analyzer.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  mar déc 3 18:59:35 CET 2002
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
#ifndef _STL_PERF_ANALYSER_HH
#define _STL_PERF_ANALYSER_HH

#include "STL_timer.hh"
#include "bench_parameter.hh"

template<class ACTION>
class STL_Perf_Analyzer{
public:  
  STL_Perf_Analyzer(unsigned long long nb_sample=DEFAULT_NB_SAMPLE):_nb_sample(nb_sample),_chronos()
  {
    MESSAGE("STL_Perf_Analyzer Ctor");
  }; 
  STL_Perf_Analyzer( const STL_Perf_Analyzer & ){
    INFOS("Copy Ctor not implemented");
    exit(0);
  };
  ~STL_Perf_Analyzer( void ){
    MESSAGE("STL_Perf_Analyzer Dtor");
  };
  
  
  inline double eval_mflops(int size)
  {

    ACTION action(size);

    _chronos.start_baseline(_nb_sample);
      
    do {

      action.initialize();
    } while (_chronos.check());

    double baseline_time=_chronos.get_time();

    _chronos.start(_nb_sample);
    do {
      action.initialize();
      action.calculate();
    } while (_chronos.check());

    double calculate_time=_chronos.get_time();

    double corrected_time=calculate_time-baseline_time;
    
    //    cout << size <<" "<<baseline_time<<" "<<calculate_time<<" "<<corrected_time<<" "<<action.nb_op_base() << endl;    
    
    return action.nb_op_base()/(corrected_time*1000000.0);
    //return action.nb_op_base()/(calculate_time*1000000.0);
    
  }
private:

  STL_Timer _chronos;
  unsigned long long _nb_sample;

  
};

  
  
#endif
