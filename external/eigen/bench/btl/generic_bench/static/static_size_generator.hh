//=====================================================
// File   :  static_size_generator.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  mar déc 3 18:59:36 CET 2002
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
#ifndef _STATIC_SIZE_GENERATOR_HH
#define _STATIC_SIZE_GENERATOR_HH
#include <vector>

using namespace std;

//recursive generation of statically defined matrix and vector sizes

template <int SIZE,template<class> class Perf_Analyzer, template<class> class Action, template<class,int> class Interface> 
struct static_size_generator{
  static void go(vector<double> & tab_sizes, vector<double> & tab_mflops)
  {
    tab_sizes.push_back(SIZE);
    std::cout << tab_sizes.back() << " \t" << std::flush;
    Perf_Analyzer<Action<Interface<REAL_TYPE,SIZE> > > perf_action;
    tab_mflops.push_back(perf_action.eval_mflops(SIZE));
    std::cout << tab_mflops.back() << " MFlops" << std::endl;
    static_size_generator<SIZE-1,Perf_Analyzer,Action,Interface>::go(tab_sizes,tab_mflops);
  };
};

//recursion end

template <template<class> class Perf_Analyzer, template<class> class Action, template<class,int> class Interface> 
struct static_size_generator<1,Perf_Analyzer,Action,Interface>{  
  static  void go(vector<double> & tab_sizes, vector<double> & tab_mflops)
  {
    tab_sizes.push_back(1);
    Perf_Analyzer<Action<Interface<REAL_TYPE,1> > > perf_action;
    tab_mflops.push_back(perf_action.eval_mflops(1));
  };
};

#endif
  
  
  
  
