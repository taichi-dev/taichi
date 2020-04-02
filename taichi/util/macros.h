#pragma once

#ifndef STR
#define _STR(...) #__VA_ARGS__
#define STR(...) _STR(__VA_ARGS__)
#endif

#ifndef CAT
#define _CAT(x, y) x##y
#define CAT(x, y) _CAT(x, y)
#endif

#ifndef CAT3
#define _CAT3(x, y, z) x##y##z
#define CAT3(x, y, z) _CAT3(x, y, z)
#endif

#ifndef EXP
#define _EXP(x) x
#define EXP(x) _EXP(x)
#endif
