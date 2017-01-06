#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <climits>
#include <limits>
#include <iostream>
#ifdef __linux__
#include <GL/glew.h>
#elif defined(__APPLE__)
#include <GL/glew.h>
#include <OpenGL/gl3.h>
#elif defined(_WIN64)
#include <GL/glew.h>
#endif
#include "Color.h"

#pragma warning(push)
#pragma warning(disable:4005)
#include <GLFW/glfw3.h>
#pragma warning(pop)
#include <cstring>
#include <cstdio>
using namespace std;
//#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" )
const int screenWidth = 1024;
const int screenHeight = 768;
const double eps = 1e-8;
const double pi = acos(-1.);
const double timeInterval = 0.1;
#define DBL_INF numeric_limits<double>::infinity()

inline float randf() {
    return 1.0f * rand() / RAND_MAX;
}

inline double sqr(double a) {
    return a * a;
}

inline double round(double number) {
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

/*
inline float invSqrt(float x)
{
    float xhalf = 0.5f*x;
    int i = *(int*)&x; // get bits for floating VALUE 
    i = 0x5f375a86- (i>>1); // gives initial guess y0
    x = *(float*)&i; // convert bits BACK to float
    x = x*(1.5f-xhalf*x*x); // Newton step, repeating increases accuracy
    x = x*(1.5f-xhalf*x*x); // Newton step, repeating increases accuracy
    x = x*(1.5f-xhalf*x*x); // Newton step, repeating increases accuracy
    return x;
}
*/

inline double invSqrt(double y)
{
    float x = (float)y;
    float xhalf = 0.5f*x;
    int i = *(int*)&x; // get bits for floating VALUE 
    i = 0x5f375a86- (i>>1); // gives initial guess y0
    x = *(float*)&i; // convert bits BACK to float
    x = x*(1.5f-xhalf*x*x); // Newton step, repeating increases accuracy
    x = x*(1.5f-xhalf*x*x); // Newton step, repeating increases accuracy
    y = x*(1.5f-xhalf*x*x); // Newton step, repeating increases accuracy
    return y;
}

inline int loopNext(int i, int n) {
    i++;
    if (i == n) return 0;
    else return i;
}