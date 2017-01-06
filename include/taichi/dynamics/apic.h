#pragma once

#include "flip_fluid.h"

TC_NAMESPACE_BEGIN

class APICFluid : public FLIPFluid {
protected:
    float apic_blend;

    virtual void initialize_solver(const Config &config);

    virtual void rasterize();

    virtual void sample_c();

    Vector2 sample_c(Vector2 &pos, Array & val);

    virtual void substep(float delta_t);
public:
    APICFluid();
};


TC_NAMESPACE_END

