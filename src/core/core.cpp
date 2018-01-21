/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

CoreState &CoreState::get_instance() {
  static CoreState state;
  return state;
}

int __trash__;

TC_NAMESPACE_END
