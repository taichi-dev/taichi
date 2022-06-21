// vim: ft=glsl
// NOLINTBEGIN(*)
#include "taichi/util/macros.h"

constexpr auto kOpenGLReductionCommon = STR(
    shared float _reduction_temp_float[gl_WorkGroupSize.x * gl_WorkGroupSize.y *
                                       gl_WorkGroupSize.z];
    shared int _reduction_temp_int[gl_WorkGroupSize.x * gl_WorkGroupSize.y *
                                   gl_WorkGroupSize.z];
    shared uint _reduction_temp_uint[gl_WorkGroupSize.x * gl_WorkGroupSize.y *
                                     gl_WorkGroupSize.z];
    float add(float a, float b) { return a + b; } int add(int a, int b) {
      return a + b;
    } uint add(uint a, uint b) { return a + b; }
\n);

#ifndef TI_INSIDE_OPENGL_CODEGEN
static_assert(false, "Do not include");
#endif

#define GENERATE_OPENGL_REDUCTION_FUNCTIONS(OP, TYPE)                          \
  constexpr auto kOpenGlReductionSource_##OP##_##TYPE =                        \
      STR(TYPE reduction_workgroup_##OP##_##TYPE(in TYPE r) {                  \
        _reduction_temp_##TYPE[gl_LocalInvocationIndex] = r;                   \
        barrier();                                                             \
        memoryBarrierShared();                                                 \
        const int group_size =                                                 \
            int(gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z); \
        const int depth = int(ceil(log2(float(group_size))));                  \
        for (int i = 0; i < depth; ++i) {                                      \
          const int radix = 1 << (i + 1);                                      \
          const int stride = 1 << i;                                           \
          const int cmp_index = int(gl_LocalInvocationIndex) + stride;         \
          if (gl_LocalInvocationIndex % radix == 0 &&                          \
              cmp_index < group_size) {                                        \
            _reduction_temp_##TYPE[gl_LocalInvocationIndex] =                  \
                OP(_reduction_temp_##TYPE[gl_LocalInvocationIndex],            \
                   _reduction_temp_##TYPE[cmp_index]);                         \
          }                                                                    \
          barrier();                                                           \
          memoryBarrierShared();                                               \
        }                                                                      \
        const TYPE result = _reduction_temp_##TYPE[0];                         \
        barrier();                                                             \
        return result;                                                         \
      });

// NOLINTEND(*)
