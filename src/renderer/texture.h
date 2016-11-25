#pragma once

#include "visualization/image_buffer.h"
#include "common/config.h"
#include "common/meta.h"

TC_NAMESPACE_BEGIN

    class Texture {
    public:
        virtual void initialize(const Config &config) {}
        virtual Vector3 sample(const Vector2 &coord) const {return Vector3(0.0f);};
    };

    TC_INTERFACE(Texture);

TC_NAMESPACE_END

