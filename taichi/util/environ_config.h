#pragma once

#include "taichi/common/util.h"
#include <string>
#include <cstdlib>

int get_environ_config(const std::string &name, int default_value = 0)
{
  char *res = std::getenv(name.c_str());
  if (res == nullptr)
    return default_value;
  return std::stoi(res);
}
