/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/texture.h>

TC_NAMESPACE_BEGIN

class MengerSponge: public Texture {
private:
    int limit;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        limit = config.get("limit", 10);
    }

    bool cut(const Vector3d &c) const {
        for (int i = 0; i < 3; i++) {
            if (c[i] < 1 / 3.0 || c[i] >= 2 / 3.0) {
                return false;
            }
        }
        return true;
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        Vector3d c = coord;
        for (int i = 0; i < limit; i++) {
            if (cut(c)) return Vector4(0.0f);
            c = fract(c * 3.0);
        }
        return Vector4(1.0f);
    }
};

TC_IMPLEMENTATION(Texture, MengerSponge, "menger");

TC_NAMESPACE_END