#pragma once

#include "taichi/common/core.h"

#include <string>

TLANG_NAMESPACE_BEGIN

enum class OffloadedTaskType : int {
#define PER_TASK_TYPE(x) x,
#include "taichi/inc/offloaded_task_type.inc.h"
#undef PER_TASK_TYPE
};

std::string offloaded_task_type_name(OffloadedTaskType tt);

TLANG_NAMESPACE_END
