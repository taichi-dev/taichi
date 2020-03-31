/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/common/interface.h"
#include "taichi/system/timer.h"

TI_NAMESPACE_BEGIN

class Benchmark : public Unit {
 protected:
  int dummy;
  int warm_up_iterations;
  int64 workload;
  bool returns_time;

  virtual void setup(){};

  virtual void iterate() = 0;

  virtual void finalize(){};

 public:
  virtual void initialize(const Config &config) override {
    warm_up_iterations = config.get("warm_up_iterations", 16);
    workload = config.get("workload", int64(1024));
    returns_time = config.get("returns_time", false);
  }

  // returns cycles per element (default) / time per element
  virtual real run(int iterations = 16) {
    setup();
    for (int i = 0; i < warm_up_iterations; i++) {
      iterate();
    }
    double start_t;
    if (returns_time)
      start_t = Time::get_time();
    else
      start_t = (double)Time::get_cycles();
    for (int i = 0; i < iterations; i++) {
      iterate();
    }
    double end_t;
    if (returns_time)
      end_t = Time::get_time();
    else
      end_t = (double)Time::get_cycles();
    real elapsed = (real)(end_t - start_t);
    finalize();
    return elapsed / (iterations * workload);
  }

  virtual bool test() const override {
    return true;
  }
};

TI_INTERFACE(Benchmark)

TI_NAMESPACE_END
