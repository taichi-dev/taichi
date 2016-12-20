#include <map>
#include <string>
#include <cstdio>
#include <cassert>
using namespace std;


#ifndef TIMER_H
#define TIMER_H

inline unsigned long long GetCPUTickCount(){ 
	return 0;
}

class Timer {
private:
	static map<string, unsigned long long> M;
public:
	static void BeginTimer(string name) {
		M[name] = GetCPUTickCount();
	}
	static void EndTimer(string name) {
		assert(M.find(name) != M.end());
		unsigned long long now = GetCPUTickCount();
		printf("Time Cost of %s is %I64u ticks\n", name.c_str(), (now - M[name]));
	}
};

#endif