#pragma once
#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

enum class EventType : int { Any = 0, Press = 1, Release = 2 };

struct Event {
  EventType tag;

  DEFINE_PROPERTY(std::string, key);
};

}  // namespace taichi::ui
