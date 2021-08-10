//=====================================================
// File   :  intel_bench_fixed_size.hh
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
#ifndef _BENCH_FIXED_SIZE_HH_
#define _BENCH_FIXED_SIZE_HH_

#include "utilities.h"
#include "function_time.hh"

template <class Action>
double bench_fixed_size(int size, unsigned long long  & nb_calc,unsigned long long & nb_init)
{
  
  Action action(size);
  
  double time_baseline=time_init(nb_init,action);

  while (time_baseline < MIN_TIME) {

    //INFOS("nb_init="<<nb_init);
    //INFOS("time_baseline="<<time_baseline);
    nb_init*=2;
    time_baseline=time_init(nb_init,action);
  }
  
  time_baseline=time_baseline/(double(nb_init));
  
  double time_action=time_calculate(nb_calc,action);
  
  while (time_action < MIN_TIME) {
    
    nb_calc*=2;
    time_action=time_calculate(nb_calc,action);
  }

  INFOS("nb_init="<<nb_init);
  INFOS("nb_calc="<<nb_calc);
  
  
  time_action=time_action/(double(nb_calc));
  
  action.check_result();
  
  time_action=time_action-time_baseline;

  return action.nb_op_base()/(time_action*1000000.0);

}

#endif
