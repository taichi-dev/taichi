#include "GL/glfw.h"
#include <map>
#include <string>
#include <cstdio>
#include <cassert>
using namespace std;


#ifndef TIMER_H
#define TIMER_H

inline unsigned long long GetCPUTickCount(){ 
	unsigned long high32=0, low32=0;
	#ifdef WIN32 // WIN32
	_asm
	{ 
		RDTSC;
		mov high32,ebx; 
		mov low32,eax; 
	} 
	#else
	__asm__ ("RDTSC" : "=a"(low32),"=d"(high32));
	#endif
	unsigned long long counter = high32;
	counter = (counter<<32) + low32;
	return low32;
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