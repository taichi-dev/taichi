#pragma once

#include <windows.h>
// Never directly include <windows.h>. That will bring you evil max/min macros.
#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif