#pragma once

#include <taichi/common/config.h>

TC_NAMESPACE_BEGIN

    class Simulator {
    public:
        virtual void initialize(const Config &config) {};
        virtual void step(float delta_t) {};
    };

TC_NAMESPACE_END
