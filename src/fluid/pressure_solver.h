#pragma once

#include <memory.h>
#include <string>
#include "visualization/image_buffer.h"
#include "common/config.h"
#include "common/interface.h"
#include "math/array_3d.h"

TC_NAMESPACE_BEGIN
    class PressureSolver3D {
    protected:
        typedef Array3D<float> Array;
    public:
        virtual void initialize(const Config &config) = 0;
        virtual void run(const Array &b, Array &x, float tolerance) = 0;
    };

    std::shared_ptr<PressureSolver3D> create_pressure_solver_3d(std::string name, const Config &config);
    //void std::shared_ptr<PressureSolver3D> create_pressure_solver_3d(std::string name, const Config &config);
TC_NAMESPACE_END
