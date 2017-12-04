/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#define CATCH_CONFIG_RUNNER
#include <taichi/common/testing.h>

TC_NAMESPACE_BEGIN

int run_tests() {
  char arg[] = "test";
  char *argv_[1];
  argv_[0] = arg;
  Catch::Session session;
  int returnCode = session.applyCommandLine(1, argv_);
  if (returnCode != 0)
    return returnCode;
  return session.run();
}

TC_NAMESPACE_END
