#pragma once

#include <taichi/common/meta.h>
#include <taichi/math/array_3d.h>

TC_NAMESPACE_BEGIN
class PressureSolver3D : public Unit {
protected:
    typedef Array3D<float> Array;
public:
    typedef unsigned char CellType;
    typedef Array3D<CellType> BCArray;
    static const CellType INTERIOR = 0;
    static const CellType DIRICHLET = 1;
    static const CellType NEUMANN = 2;

    virtual void run(const Array &b, Array &x, float tolerance) {};
    virtual void set_boundary_condition(const BCArray &boundary) {};
};

TC_INTERFACE(PressureSolver3D);

TC_NAMESPACE_END
