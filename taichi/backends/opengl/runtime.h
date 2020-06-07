#define MAX_MESSAGES (1024 * 4)  // * 4 * 32 = 512 KB
#define MSG_SIZE 32
// 2 left for the `type` bitmap, 1 left for the contents-count
#define MAX_CONTENTS_PER_MSG (MSG_SIZE - 3)

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
  int msg_buf[MAX_MESSAGES * MSG_SIZE];
} __attribute__((packed));

#endif
