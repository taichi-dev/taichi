/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/common/core.h"

#if defined(TI_PLATFORM_WINDOWS)
#include "taichi/platform/windows/windows.h"
#else
// Mac and Linux
#include <unistd.h>
#endif

TI_NAMESPACE_BEGIN

extern "C" {
#if defined(TI_PLATFORM_LINUX) && defined(TI_ARCH_x64)
// Avoid dependency on higher glibc versions such as 2.27 or 2.29
// Related issue: https://github.com/taichi-dev/taichi/issues/3174
// log2f is used by a third party .a file, so we have to define a wrapper.
// https://stackoverflow.com/questions/8823267/linking-against-older-symbol-version-in-a-so-file
// The wrapper should be linked using target_link_libraries in TaichiCore.cmake
__asm__(".symver log2f,log2f@GLIBC_2.2.5");
float __wrap_log2f(float x) {
  return log2f(x);
}
// The following are offending symbols using higher GLIBC versions
// TODO currently commented out due to failing Vulkan tests
//__asm__(".symver log2,log2@GLIBC_2.2.5");
// float __wrap_log2(float x) {
//  return log2(x);
//}
//__asm__(".symver exp,exp@GLIBC_2.2.5");
// float __wrap_exp(float x) {
//  return exp(x);
//}
//__asm__(".symver log,log@GLIBC_2.2.5");
// float __wrap_log(float x) {
//  return log(x);
//}
//__asm__(".symver pow,pow@GLIBC_2.2.5");
// float __wrap_pow(float x, float y) {
//  return pow(x, y);
//}
#endif
}

TI_NAMESPACE_END
