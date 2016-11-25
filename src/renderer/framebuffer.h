#pragma once

#include "common/config.h"
#include "common/meta.h"

TC_NAMESPACE_BEGIN

    class Framebuffer {
    public:
        virtual void initialize(const Config &config) {};
    protected:
    };

    TC_INTERFACE(Framebuffer);

TC_NAMESPACE_END

