#pragma once

#include "geometry_primitives.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include "common/meta.h"

TC_NAMESPACE_BEGIN

    class RayIntersection {
    public:
        virtual void clear() = 0;

        virtual void build() = 0;

        virtual void query(Ray &ray) = 0;

        virtual bool occlude(Ray &ray) = 0;

        virtual void add_triangle(Triangle &triangle) = 0;
    };

    TC_INTERFACE(RayIntersection);

TC_NAMESPACE_END
