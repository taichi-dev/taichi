//=====================================================
// File   :  mixed_perf_analyzer.hh
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
#ifndef _MIXED_PERF_ANALYSER_HH
#define _MIXED_PERF_ANALYSER_HH

#include "x86_perf_analyzer.hh"
#include "portable_perf_analyzer.hh"

// choose portable perf analyzer for long calculations and x86 analyser for short ones


template<class Action>
class Mixed_Perf_Analyzer{
  
public:  
  Mixed_Perf_Analyzer( void ):_x86pa(),_ppa(),_use_ppa(true)
  {
    MESSAGE("Mixed_Perf_Analyzer Ctor");
  }; 
  Mixed_Perf_Analyzer( const Mixed_Perf_Analyzer & ){
    INFOS("Copy Ctor not implemented");
    exit(0);
  };
  ~Mixed_Perf_Analyzer( void ){
    MESSAGE("Mixed_Perf_Analyzer Dtor");
  };
    
  
  inline double eval_mflops(int size)
  {

    double result=0.0;
    if (_use_ppa){      
      result=_ppa.eval_mflops(size);
      if (_ppa.get_nb_calc()>DEFAULT_NB_SAMPLE){_use_ppa=false;}      
    }
    else{      
      result=_x86pa.eval_mflops(size);
    }

    return result;
  }

private:

  Portable_Perf_Analyzer<Action> _ppa;
  X86_Perf_Analyzer<Action> _x86pa;
  bool _use_ppa;

};

#endif

  
    
  
