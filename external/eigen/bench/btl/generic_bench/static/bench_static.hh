//=====================================================
// File   :  bench_static.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  lun sep 30 14:23:16 CEST 2002
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
#ifndef BENCH_STATIC_HH
#define BENCH_STATIC_HH

#include "btl.hh"
#include "bench_parameter.hh"
#include <iostream>
#include "utilities.h"
#include "xy_file.hh"
#include "static/static_size_generator.hh"
#include "timers/portable_perf_analyzer.hh"
// #include "timers/mixed_perf_analyzer.hh"
// #include "timers/x86_perf_analyzer.hh"

using namespace std;


template <template<class> class Perf_Analyzer, template<class> class Action, template<class,int> class Interface>
BTL_DONT_INLINE  void bench_static(void)
{
  if (BtlConfig::skipAction(Action<Interface<REAL_TYPE,10> >::name()))
    return;

  string filename = "bench_" + Action<Interface<REAL_TYPE,10> >::name() + ".dat";

  INFOS("starting " << filename);

  const int max_size = TINY_MV_MAX_SIZE;

  std::vector<double> tab_mflops;
  std::vector<double> tab_sizes;

  static_size_generator<max_size,Perf_Analyzer,Action,Interface>::go(tab_sizes,tab_mflops);

  dump_xy_file(tab_sizes,tab_mflops,filename);
}

// default Perf Analyzer
template <template<class> class Action, template<class,int> class Interface>
BTL_DONT_INLINE  void bench_static(void)
{
  bench_static<Portable_Perf_Analyzer,Action,Interface>();
  //bench_static<Mixed_Perf_Analyzer,Action,Interface>();
  //bench_static<X86_Perf_Analyzer,Action,Interface>();
}

#endif















