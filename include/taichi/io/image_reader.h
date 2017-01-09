#pragma once

#include <taichi/math/linalg.h>
#include <taichi/image/image_buffer.h>
#include <taichi/common/meta.h>

TC_NAMESPACE_BEGIN

class ImageReader : public Unit {
public:
    ImageReader() {}
    virtual Array2D<Vector4> read(const std::string &filepath) {
        return Array2D<Vector4>(0, 0);
    }
};

TC_INTERFACE(ImageReader);

TC_NAMESPACE_END

