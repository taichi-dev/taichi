#define MAX_PRINT_ENTRIES (1024 * 4) // * 2 * 4 * 32 = 1 MB
#define MAX_CONTENTS_PER_PRINT 32

#ifdef __GLSL__
// clang-format off

#include "taichi/util/macros.h"
STR(
layout(shared, binding = 6) buffer runtime {
  int _rand_state_;
  int _msg_count_;
  int _mesg_i32_[];
};
)

// clang-format on
#else

struct GLSLRuntime {
  int rand_state;
  int msg_count;
  int msg_buf[MAX_PRINT_ENTRIES * MAX_CONTENTS_PER_PRINT * 2];
} __attribute__((packed));

#endif
