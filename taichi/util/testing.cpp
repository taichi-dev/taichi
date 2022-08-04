/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#define CATCH_CONFIG_RUNNER
#include "taichi/util/testing.h"

namespace taichi {

int run_tests(std::vector<std::string> argv) {
  char arg[] = "test";
  std::vector<char *> argv_(1 + argv.size(), nullptr);
  argv_[0] = arg;
  for (int i = 0; i < (int)argv.size(); i++) {
    argv_[i + 1] = const_cast<char *>(argv[i].c_str());
  }
  Catch::Session session;
  int returnCode = session.applyCommandLine(1 + argv.size(), argv_.data());
  if (returnCode != 0)
    return returnCode;
  return session.run();
}

}  // namespace taichi
