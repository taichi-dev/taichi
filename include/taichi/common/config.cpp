/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <map>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <typeinfo>

#include <taichi/common/string_utils.h>
#include <taichi/common/config.h>
#include <taichi/common/util.h>
#include <taichi/common/asset_manager.h>
#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

template <>
real Config::get<real>(std::string key) const {
  return this->get_real(key);
}

template <>
int Config::get<int>(std::string key) const {
  return this->get_int(key);
}

template <>
bool Config::get<bool>(std::string key) const {
  return this->get_bool(key);
}

template <>
std::string Config::get<std::string>(std::string key) const {
  return this->get_string(key);
}

TC_NAMESPACE_END
