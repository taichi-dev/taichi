#include "exports.h"

#include <cstdio>
#include <cstring>

int ticore_hello_world(const char *extra_msg) {
  std::printf("Hello World! %s\n", extra_msg);
  return std::strlen(extra_msg);
}
