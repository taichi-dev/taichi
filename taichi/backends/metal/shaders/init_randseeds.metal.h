#include "taichi/backends/metal/shaders/prolog.h"

#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define METAL_BEGIN_INIT_RANDSEEDS_DEF \
  constexpr auto kMetalInitRandseedsSourceCode =
#define METAL_END_INIT_RANDSEEDS_DEF ;
#else
#define METAL_BEGIN_INIT_RANDSEEDS_DEF
#define METAL_END_INIT_RANDSEEDS_DEF
#endif  // TI_METAL_NESTED_INCLUDE

#else

#define METAL_BEGIN_INIT_RANDSEEDS_DEF
#define METAL_END_INIT_RANDSEEDS_DEF

#endif  // TI_INSIDE_METAL_CODEGEN

METAL_BEGIN_INIT_RANDSEEDS_DEF
STR([[maybe_unused]] void mtl_init_random_seeds(
    device uint32_t *rand_seeds,
    const uint thread_position_in_grid,
    const uint threads_per_grid) {
  for (int ii = thread_position_in_grid; ii < threads_per_grid; ++ii) {
    rand_seeds[ii] += ii;
  }
})
METAL_END_INIT_RANDSEEDS_DEF