//=====================================================
// File   :  x86_perf_analyzer.hh
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
#ifndef _X86_PERF_ANALYSER_HH
#define _X86_PERF_ANALYSER_HH

#include "x86_timer.hh"
#include "bench_parameter.hh"

template<class ACTION>
class X86_Perf_Analyzer{
public:
  X86_Perf_Analyzer( unsigned long long nb_sample=DEFAULT_NB_SAMPLE):_nb_sample(nb_sample),_chronos()
  {
    MESSAGE("X86_Perf_Analyzer Ctor");
    _chronos.find_frequency();
  };
  X86_Perf_Analyzer( const X86_Perf_Analyzer & ){
    INFOS("Copy Ctor not implemented");
    exit(0);
  };
  ~X86_Perf_Analyzer( void ){
    MESSAGE("X86_Perf_Analyzer Dtor");
  };


  inline double eval_mflops(int size)
  {

    ACTION action(size);

    int nb_loop=5;
    double calculate_time=0.0;
    double baseline_time=0.0;

    for (int j=0 ; j < nb_loop ; j++){

      _chronos.clear();

      for(int i=0 ; i < _nb_sample  ; i++)
      {
        _chronos.start();
        action.initialize();
        action.calculate();
        _chronos.stop();
        _chronos.add_get_click();
      }

      calculate_time += double(_chronos.get_shortest_clicks())/_chronos.frequency();

      if (j==0) action.check_result();

      _chronos.clear();

      for(int i=0 ; i < _nb_sample  ; i++)
      {
        _chronos.start();
        action.initialize();
        _chronos.stop();
        _chronos.add_get_click();

      }

      baseline_time+=double(_chronos.get_shortest_clicks())/_chronos.frequency();

    }

    double corrected_time = (calculate_time-baseline_time)/double(nb_loop);


//     INFOS("_nb_sample="<<_nb_sample);
//     INFOS("baseline_time="<<baseline_time);
//     INFOS("calculate_time="<<calculate_time);
//     INFOS("corrected_time="<<corrected_time);

//    cout << size <<" "<<baseline_time<<" "<<calculate_time<<" "<<corrected_time<<" "<<action.nb_op_base() << endl;

    return action.nb_op_base()/(corrected_time*1000000.0);
    //return action.nb_op_base()/(calculate_time*1000000.0);
  }

private:

  X86_Timer _chronos;
  unsigned long long _nb_sample;


};



#endif
