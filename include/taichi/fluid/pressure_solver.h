#pragma once

#include <taichi/common/meta.h>
#include <taichi/math/array_3d.h>

TC_NAMESPACE_BEGIN
class PressureSolver3D {
protected:
	typedef Array3D<float> Array;
public:
	virtual void initialize(const Config &config) = 0;
	virtual void run(const Array &b, Array &x, float tolerance) = 0;
};

TC_INTERFACE(PressureSolver3D);

TC_NAMESPACE_END
