#pragma once

#include <taichi/common/meta.h>
#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

class SDF : public Unit {
public:

    virtual void initialize(const Config &config) {}

    virtual real eval(const Vector3 &p) const { return 1; }

};

TC_INTERFACE(SDF);

TC_NAMESPACE_END

