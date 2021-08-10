//=====================================================
// File   :  portable_perf_analyzer.hh
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
#ifndef _PORTABLE_PERF_ANALYZER_HH
#define _PORTABLE_PERF_ANALYZER_HH

#include "utilities.h"
#include "timers/portable_timer.hh"

template <class Action>
class Portable_Perf_Analyzer{
public:
  Portable_Perf_Analyzer( void ):_nb_calc(1),_nb_init(1),_chronos(){
    MESSAGE("Portable_Perf_Analyzer Ctor");
  };
  Portable_Perf_Analyzer( const Portable_Perf_Analyzer & ){
    INFOS("Copy Ctor not implemented");
    exit(0);
  };
  ~Portable_Perf_Analyzer( void ){
    MESSAGE("Portable_Perf_Analyzer Dtor");
  };



  inline double eval_mflops(int size)
  {

    Action action(size);

//     double time_baseline = time_init(action);
//     while (time_baseline < MIN_TIME_INIT)
//     {
//       _nb_init *= 2;
//       time_baseline = time_init(action);
//     }
//
//     // optimize
//     for (int i=1; i<NB_TRIES; ++i)
//       time_baseline = std::min(time_baseline, time_init(action));
//
//     time_baseline = time_baseline/(double(_nb_init));

    double time_action = time_calculate(action);
    while (time_action < MIN_TIME)
    {
      _nb_calc *= 2;
      time_action = time_calculate(action);
    }

    // optimize
    for (int i=1; i<NB_TRIES; ++i)
      time_action = std::min(time_action, time_calculate(action));

//     INFOS("size="<<size);
//     INFOS("_nb_init="<<_nb_init);
//     INFOS("_nb_calc="<<_nb_calc);

    time_action = time_action / (double(_nb_calc));

    action.check_result();


    double time_baseline = time_init(action);
    for (int i=1; i<NB_TRIES; ++i)
      time_baseline = std::min(time_baseline, time_init(action));
    time_baseline = time_baseline/(double(_nb_init));



//     INFOS("time_baseline="<<time_baseline);
//     INFOS("time_action="<<time_action);

    time_action = time_action - time_baseline;

//     INFOS("time_corrected="<<time_action);

    return action.nb_op_base()/(time_action*1000000.0);
  }

  inline double time_init(Action & action)
  {
    // time measurement
    _chronos.start();
    for (int ii=0; ii<_nb_init; ii++)
      action.initialize();
    _chronos.stop();
    return _chronos.user_time();
  }


  inline double time_calculate(Action & action)
  {
    // time measurement
    _chronos.start();
    for (int ii=0;ii<_nb_calc;ii++)
    {
      action.initialize();
      action.calculate();
    }
    _chronos.stop();
    return _chronos.user_time();
  }

  unsigned long long get_nb_calc( void )
  {
    return _nb_calc;
  }


private:
  unsigned long long _nb_calc;
  unsigned long long _nb_init;
  Portable_Timer _chronos;

};

#endif //_PORTABLE_PERF_ANALYZER_HH
