#pragma once

#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

#define SOLVER_EIGEN 1
#define SOLVER_NAIVE 0

extern float volume_control_p;
extern float volume_control_i;
extern const bool enable_volume_control;

TC_NAMESPACE_END

