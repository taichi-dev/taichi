#pragma once

#include "flip_liquid.h"

TC_NAMESPACE_BEGIN

class APICLiquid : public FLIPLiquid {
protected:
    real apic_blend;

    virtual void initialize_solver(const Config &config);

    virtual void rasterize();

    virtual void sample_c();

    Vector2 sample_c(Vector2 &pos, Array & val);

    virtual void substep(real delta_t);
public:
    APICLiquid();
};


TC_NAMESPACE_END

