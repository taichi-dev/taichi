//=====================================================
// File   :  bench.hh
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
#ifndef BENCH_HH
#define BENCH_HH

#include "btl.hh"
#include "bench_parameter.hh"
#include <iostream>
#include "utilities.h"
#include "size_lin_log.hh"
#include "xy_file.hh"
#include <vector>
#include <string>
#include "timers/portable_perf_analyzer.hh"
// #include "timers/mixed_perf_analyzer.hh"
// #include "timers/x86_perf_analyzer.hh"
// #include "timers/STL_perf_analyzer.hh"
#ifdef HAVE_MKL
extern "C" void cblas_saxpy(const int, const float, const float*, const int, float *, const int);
#endif
using namespace std;

template <template<class> class Perf_Analyzer, class Action>
BTL_DONT_INLINE void bench( int size_min, int size_max, int nb_point )
{
  if (BtlConfig::skipAction(Action::name()))
    return;

  string filename="bench_"+Action::name()+".dat";

  INFOS("starting " <<filename);

  // utilities

  std::vector<double> tab_mflops(nb_point);
  std::vector<int> tab_sizes(nb_point);

  // matrices and vector size calculations
  size_lin_log(nb_point,size_min,size_max,tab_sizes);

  std::vector<int> oldSizes;
  std::vector<double> oldFlops;
  bool hasOldResults = read_xy_file(filename, oldSizes, oldFlops, true);
  int oldi = oldSizes.size() - 1;

  // loop on matrix size
  Perf_Analyzer<Action> perf_action;
  for (int i=nb_point-1;i>=0;i--)
  {
    //INFOS("size=" <<tab_sizes[i]<<"   ("<<nb_point-i<<"/"<<nb_point<<")");
    std::cout << " " << "size = " << tab_sizes[i] << "  " << std::flush;

    BTL_DISABLE_SSE_EXCEPTIONS();
    #ifdef HAVE_MKL
    {
      float dummy;
      cblas_saxpy(1,0,&dummy,1,&dummy,1);
    }
    #endif

    tab_mflops[i] = perf_action.eval_mflops(tab_sizes[i]);
    std::cout << tab_mflops[i];
    
    if (hasOldResults)
    {
      while (oldi>=0 && oldSizes[oldi]>tab_sizes[i])
        --oldi;
      if (oldi>=0 && oldSizes[oldi]==tab_sizes[i])
      {
        if (oldFlops[oldi]<tab_mflops[i])
          std::cout << "\t > ";
        else
          std::cout << "\t < ";
        std::cout << oldFlops[oldi];
      }
      --oldi;
    }
    std::cout << " MFlops    (" << nb_point-i << "/" << nb_point << ")" << std::endl;
  }

  if (!BtlConfig::Instance.overwriteResults)
  {
    if (hasOldResults)
    {
      // merge the two data
      std::vector<int> newSizes;
      std::vector<double> newFlops;
      unsigned int i=0;
      unsigned int j=0;
      while (i<tab_sizes.size() && j<oldSizes.size())
      {
        if (tab_sizes[i] == oldSizes[j])
        {
          newSizes.push_back(tab_sizes[i]);
          newFlops.push_back(std::max(tab_mflops[i], oldFlops[j]));
          ++i;
          ++j;
        }
        else if (tab_sizes[i] < oldSizes[j])
        {
          newSizes.push_back(tab_sizes[i]);
          newFlops.push_back(tab_mflops[i]);
          ++i;
        }
        else
        {
          newSizes.push_back(oldSizes[j]);
          newFlops.push_back(oldFlops[j]);
          ++j;
        }
      }
      while (i<tab_sizes.size())
      {
        newSizes.push_back(tab_sizes[i]);
        newFlops.push_back(tab_mflops[i]);
        ++i;
      }
      while (j<oldSizes.size())
      {
        newSizes.push_back(oldSizes[j]);
        newFlops.push_back(oldFlops[j]);
        ++j;
      }
      tab_mflops = newFlops;
      tab_sizes = newSizes;
    }
  }

  // dump the result in a file  :
  dump_xy_file(tab_sizes,tab_mflops,filename);

}

// default Perf Analyzer

template <class Action>
BTL_DONT_INLINE void bench( int size_min, int size_max, int nb_point ){

  // if the rdtsc is not available :
  bench<Portable_Perf_Analyzer,Action>(size_min,size_max,nb_point);
  // if the rdtsc is available :
//    bench<Mixed_Perf_Analyzer,Action>(size_min,size_max,nb_point);


  // Only for small problem size. Otherwize it will be too long
//   bench<X86_Perf_Analyzer,Action>(size_min,size_max,nb_point);
//   bench<STL_Perf_Analyzer,Action>(size_min,size_max,nb_point);

}

#endif
