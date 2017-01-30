#pragma once

#include "taichi/dynamics/fluid2d/euler_fluid.h"

TC_NAMESPACE_BEGIN

class EulerSmoke : public EulerFluid {
protected:
    //int advection_order;


    //virtual void advect(real delta_t);
    virtual void emit(real delta_t);

    virtual void substep(real delta_t);

    Array temperature;
    real buoyancy_alpha;
    real buoyancy_beta;

public:
    EulerSmoke() {}

    virtual void apply_external_forces(real delta_t);

    virtual void initialize(const Config &config);

};

TC_NAMESPACE_END

