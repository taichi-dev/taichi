//=====================================================
// File   :  STL_Timer.hh
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
// STL Timer Class. Adapted (L.P.) from the timer class by Musser et Al
// described int the Book : STL Tutorial and reference guide.
// Define a timer class for analyzing algorithm performance.
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;

class STL_Timer {
public:
  STL_Timer(){ baseline = false; };  // Default constructor
  // Start a series of r trials:
  void start(unsigned int r){
    reps = r;
    count = 0;
    iterations.clear();
    iterations.reserve(reps);
    initial = time(0);
  };
  // Start a series of r trials to determine baseline time:
  void start_baseline(unsigned int r)
  {
    baseline = true;
    start(r);
  }
  // Returns true if the trials have been completed, else false
  bool check()
  {
    ++count;
    final = time(0);
    if (initial < final) {
      iterations.push_back(count);  
      initial = final;
      count = 0;
    }
    return (iterations.size() < reps);
  };
  // Returns the results for external use
  double get_time( void )
  {
    sort(iterations.begin(), iterations.end());
    return 1.0/iterations[reps/2];
  };
private:
  unsigned int reps;  // Number of trials
  // For storing loop iterations of a trial
  vector<long> iterations;
  // For saving initial and final times of a trial
  time_t initial, final;
  // For counting loop iterations of a trial
  unsigned long count;
  // true if this is a baseline computation, false otherwise
  bool baseline;
  // For recording the baseline time 
  double baseline_time;
};

