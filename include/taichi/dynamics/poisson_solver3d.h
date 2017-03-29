/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/meta.h>
#include <taichi/math/array_3d.h>

TC_NAMESPACE_BEGIN
class PoissonSolver3D : public Unit {
protected:
    typedef Array3D<float> Array;
    int maximum_iterations;
public:
    typedef unsigned char CellType;
    typedef Array3D<CellType> BCArray;
    static const CellType INTERIOR = 0;
    static const CellType DIRICHLET = 1;
    static const CellType NEUMANN = 2;

    void initialize(const Config &config);
    virtual void run(const Array &b, Array &x, float tolerance) {};
    virtual void set_boundary_condition(const BCArray &boundary) {};
};

TC_INTERFACE(PoissonSolver3D);

TC_NAMESPACE_END
