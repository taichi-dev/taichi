/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/meta.h>
#include <taichi/math/array_2d.h>

//TODO: I copy and paste the code of PoissonSolver3D for now (since I'm in a rush),
// but we need to unify 2D/3D poisson solve with templates

TC_NAMESPACE_BEGIN
class PoissonSolver2D : public Unit {
protected:
    typedef Array2D<float> Array;
public:
    typedef unsigned char CellType;
    typedef Array2D<CellType> BCArray;
    static const CellType INTERIOR = 0;
    static const CellType DIRICHLET = 1;
    static const CellType NEUMANN = 2;

    virtual void run(const Array &b, Array &x, float tolerance) {};
    virtual void set_boundary_condition(const BCArray &boundary) {};
};

TC_INTERFACE(PoissonSolver2D);

TC_NAMESPACE_END
