#pragma once

#include <string>
#include "taichi/common/core.h"

TLANG_NAMESPACE_BEGIN

enum OffloadedTaskType : int {
#define PER_TASK_TYPE(x) x,
#include "taichi/inc/offloaded_task_type.inc.h"
#undef PER_TASK_TYPE
};

std::string offloaded_task_type_name(OffloadedTaskType tt);

TLANG_NAMESPACE_END
