#pragma once

#include <taichi/hdr/tone_mapper.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(ToneMapper, "tone_mapper")

class GradientDomainTMO final : public ToneMapper {
public:
    void initialize(const Config &config) {
        
    }

    Array2D<Vector3> apply(const Array2D<Vector3> &inp) {
        return inp;
    }
};

TC_IMPLEMENTATION(ToneMapper, GradientDomainTMO, "gradient")

TC_NAMESPACE_END