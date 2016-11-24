#pragma once

#include "visualization/image_buffer.h"

TC_NAMESPACE_BEGIN

    class AbstractTexture {
    public:
        virtual Vector3 sample(const Vector2 &coord) const = 0;
    };

    class ConstantTexture : public AbstractTexture {
    private:
        Vector3 val;
    public:
        ConstantTexture(const Vector3 &val) : val(val) {}
        Vector3 sample(const Vector2 &coord) const override {
            return val;
        }
    };

    class ImageTexture : public AbstractTexture {
    protected:
        ImageBuffer<Vector3> image;
    public:
        ImageTexture(const string &filename) {
            image.load(filename);
        }
        Vector3 sample(const Vector2 &coord) const override  {
            return image.sample_relative_coord(coord);
        }
    };

TC_NAMESPACE_END

