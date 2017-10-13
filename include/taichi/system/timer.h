/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#ifndef _WIN64

#include <sys/time.h>

#else
#pragma warning(push)
#pragma warning(disable : 4005)
#include <windows.h>
#pragma warning(pop)
#endif

#include <string>
#include <cstdio>
#include <map>
#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

// #define TIME(x) {Time::Timer _(#x "@" __FILENAME__ "[" __LINE__"]"); x;}
#define TIME(x)                                                      \
  {                                                                  \
    char timer_name[1000];                                           \
    sprintf_s(timer_name, "%s[%d]: %s", __FILENAME__, __LINE__, #x); \
    Time::Timer _(timer_name);                                       \
    x;                                                               \
  }

#include <stdint.h>

class Time {
 public:
  static double get_time();

  static uint64 get_cycles();

  static void usleep(double us);

  class Timer {
    static std::map<std::string, std::pair<double, int>> memo;

   protected:
    std::string name;
    double start_time;

    virtual double get_time();

    virtual void print_record(const char *left, double elapsed, double average);

    void output();

    bool have_output;

   public:
    Timer(std::string name);

    Timer() {
    }

    virtual ~Timer() {
      output();
    }
  };

  class TickTimer : public Timer {
   protected:
    double get_time();

    void print_record(const char *left, double elapsed, double average);

   public:
    TickTimer(std::string name);

    ~TickTimer() {
      output();
    }
  };

  class FPSCounter {
   public:
    static void count(std::string name) {
      if (last_refresh.find(name) == last_refresh.end()) {
        last_refresh[name] = get_time();
        counter[name] = 0;
      }
      counter[name]++;
      double current_time = get_time();
      if (current_time > 1 + last_refresh[name]) {
        last_refresh[name] = last_refresh[name] + 1;
        printf("FPS [%s]: %d\n", name.c_str(), counter[name]);
        counter[name] = 0;
      }
    }

   private:
    static std::map<std::string, double> last_refresh;
    static std::map<std::string, int> counter;
  };
};

TC_NAMESPACE_END
