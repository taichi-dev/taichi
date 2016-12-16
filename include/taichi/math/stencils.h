#pragma once
#include "math_util.h"
#include "linalg.h"
#include <vector>

TC_NAMESPACE_BEGIN

static const Vector2i neighbour4[] { Vector2i(0, 1), Vector2i(0, -1), Vector2i(-1, 0), Vector2i(1, 0) };
#define ROW(x) Vector2i((x), -1), Vector2i(x, 0), Vector2i(x, 1), Vector2i(x, 2)
static const Vector2i rasterize_support2[] { 
	ROW(-1),
	ROW( 0),
	ROW( 1),
	ROW( 2)
};
#undef ROW

TC_NAMESPACE_END

