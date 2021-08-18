#pragma once
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

enum class EventType : int { Any = 0, Press = 1, Release = 2 };

struct Event {
  EventType tag;

  DEFINE_PROPERTY(std::string, key);
};

TI_UI_NAMESPACE_END
