/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/texture.h>
#include <taichi/visual/scene_geometry.h>
#include <taichi/visualization/image_buffer.h>
#include <taichi/math/array_3d.h>
#include <taichi/common/asset_manager.h>

TC_NAMESPACE_BEGIN

class ConstantTexture final : public Texture {
private:
    Vector4 val;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        val = config.get_vec4("value");
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        return val;
    }
};

TC_IMPLEMENTATION(Texture, ConstantTexture, "const");

class Array3DTexture : public Texture {
protected:
    Array3D<Vector4> arr;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        arr = *config.get_ptr<Array3D<Vector4>>("array_ptr");
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        return arr.sample_relative_coord(coord);
    }
};

TC_IMPLEMENTATION(Texture, Array3DTexture, "array3d");

class ImageTexture : public Texture {
protected:
    Array2D<Vector4> image;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        image.load(config.get_string("filename"));
    }

    virtual Vector4 sample(const Vector3 &coord_) const override {
        Vector2 coord(coord_.x - floor(coord_.x), coord_.y - floor(coord_.y));
        return image.sample_relative_coord(coord);
    }
};

TC_IMPLEMENTATION(Texture, ImageTexture, "image");

class TextTexture : public Texture {
protected:
    Array2D<Vector4> image;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        int width = config.get_int("width");
        int height = config.get_int("height");
        std::string font_file_fn = config.get_string("font_file");
        std::string content = config.get_string("content");
        real size = config.get_real("size");
        int dx = config.get_int("dx");
        int dy = config.get_int("dy");
        image.initialize(width, height);
        image.write_text(font_file_fn, content, size, dx, dy);
    }

    virtual Vector4 sample(const Vector3 &coord_) const override {
        Vector2 coord(coord_.x - floor(coord_.x), coord_.y - floor(coord_.y));
        return image.sample_relative_coord(coord);
    }
};

TC_IMPLEMENTATION(Texture, TextTexture, "text");

class RectTexture : public Texture {
protected:
    Vector3 bounds;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        bounds = config.get_vec3("bounds") * 0.5f;
    }

    bool inside(const Vector3 &coord) const {
        Vector3 c = coord - Vector3(0.5f);
        return
                std::abs(c.x) < bounds.x &&
                std::abs(c.y) < bounds.y &&
                std::abs(c.z) < bounds.z;
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        return Vector4(inside(coord) ? 1.0f : 0.0f);
    }
};

TC_IMPLEMENTATION(Texture, RectTexture, "rect");

class RingTexture : public Texture {
protected:
    real inner, outer;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        inner = config.get("inner", 0.0f) / 2.0f;
        outer = config.get("outer", 1.0f) / 2.0f;
    }

    static bool inside(Vector2 p, Vector2 c, real r) {
        return (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) <= r * r;
    }

    virtual Vector4 sample(const Vector2 &coord) const override {
        return Vector4(
                inside(coord, Vector2(0.5f, 0.5f), outer) &&
                !inside(coord, Vector2(0.5f, 0.5f), inner) ? 1.0f : 0.0f);
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        return sample(Vector2(coord.x, coord.y));
    }
};

TC_IMPLEMENTATION(Texture, RingTexture, "ring");

class TaichiTexture : public Texture {
protected:
    real scale;
    real rotation_c, rotation_s;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        scale = config.get("scale", 1.0f);
        real rotation = config.get("rotation", 0.0f);
        rotation_c = std::cos(rotation);
        rotation_s = std::sin(rotation);
    }

    static bool inside(Vector2 p, Vector2 c, real r) {
        return (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) <= r * r;
    }

    static bool inside_left(Vector2 p, Vector2 c, real r) {
        return inside(p, c, r) && p.x < c.x;
    }

    static bool inside_right(Vector2 p, Vector2 c, real r) {
        return inside(p, c, r) && p.x >= c.x;
    }

    bool is_white(Vector2 p_) const {
        p_ -= Vector2(0.5f);
        Vector2 p(p_.x * rotation_c + p_.y * rotation_s, -p_.x * rotation_s + p_.y * rotation_c);
        p += Vector2(0.5f);
        if (!inside(p, Vector2(0.50f, 0.50f), 0.5f)) {
            return true;
        }
        if (!inside(p, Vector2(0.50f, 0.50f), 0.5f * scale)) {
            return false;
        }
        p = Vector2(0.5f) + (p - Vector2(0.5f)) * (1.0f / scale);
        if (inside(p, Vector2(0.50f, 0.25f), 0.08f)) {
            return true;
        }
        if (inside(p, Vector2(0.50f, 0.75f), 0.08f)) {
            return false;
        }
        if (inside(p, Vector2(0.50f, 0.25f), 0.25f)) {
            return false;
        }
        if (inside(p, Vector2(0.50f, 0.75f), 0.25f)) {
            return true;
        }
        if (p.x < 0.5f) {
            return true;
        } else {
            return false;
        }
    }

    virtual Vector4 sample(const Vector2 &coord) const override {
        return Vector4(is_white(coord) ? 1.0f : 0.0f);
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        return sample(Vector2(coord.x, coord.y));
    }
};

TC_IMPLEMENTATION(Texture, TaichiTexture, "taichi");

class CheckerboardTexture : public Texture {
protected:
    real repeat_u, repeat_v, repeat_w;
    std::shared_ptr<Texture> tex1, tex2;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        tex1 = AssetManager::get_asset<Texture>(config.get_int("tex1"));
        tex2 = AssetManager::get_asset<Texture>(config.get_int("tex2"));
        repeat_u = config.get_real("repeat_u");
        repeat_v = config.get_real("repeat_v");
        repeat_w = config.get("repeat_w", 1.0f);
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        int p = (int) floor(coord.x * repeat_u), q = (int) floor(coord.y * repeat_v),
                r = (int) floor(coord.z * repeat_w);
        return ((p + q + r) % 2 == 0 ? tex1 : tex2)->sample(coord);
    }
};

TC_IMPLEMENTATION(Texture, CheckerboardTexture, "checkerboard");

class UVTexture : public Texture {
protected:
    real coeff_u, coeff_v;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        coeff_u = config.get_real("coeff_u");
        coeff_v = config.get_real("coeff_v");
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        return Vector4(coeff_u * coord.x + coeff_v * coord.y);
    }
};

TC_IMPLEMENTATION(Texture, UVTexture, "uv");

class SphereTexture : public Texture {
protected:
    Vector3 center;
    real radius;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        center = config.get_vec3("center");
        radius = config.get_real("radius");
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        return Vector4(real(bool(glm::length(coord - center) < radius)));
    }
};

TC_IMPLEMENTATION(Texture, SphereTexture, "sphere");

class SlicedTexture : public Texture {
protected:
    Vector4 base;
    Vector4 increment;
    Vector4 steps;
public:
    void initialize(const Config &config) override {
        Texture::initialize(config);
        steps = config.get_vec4("steps");
        base = config.get_vec4("base");
        increment = config.get_vec4("increment");
    }

    virtual Vector4 sample(const Vector3 &coord_) const override {
        Vector4 coord(coord_, 0.0f);
        return base + floor(coord * steps) * increment;
    }
};

TC_IMPLEMENTATION(Texture, SlicedTexture, "sliced");

class MeshTexture : public Texture {
protected:
    Array3D<char> arr;
    Vector3i resolution;
    int resolution_x;
    int resolution_y;
    int resolution_z;
public:
    // parameter name for mesh path is 'filename'
    void initialize(const Config &config) override {
        Mesh mesh;
        mesh.initialize(config);
        Scene scene;
        scene.add_mesh(std::make_shared<Mesh>(mesh));
        scene.finalize_geometry();
        auto ray_intersection = create_instance<RayIntersection>("embree");
        std::shared_ptr<SceneGeometry> scene_geometry(new SceneGeometry(std::make_shared<Scene>(scene), ray_intersection));
        resolution = config.get_vec3i("resolution");
        arr = Array3D<char>(resolution.x, resolution.y, resolution.z);
        BoundingBox bb = mesh.get_bounding_box();
        for (int i = 0; i < resolution.x; ++i)
            for (int j = 0; j < resolution.y; ++j) {
                real x = lerp((0.5f + i) / resolution.x, bb.lower_boundary.x, bb.upper_boundary.x);
                real y = lerp((0.5f + j) / resolution.y, bb.lower_boundary.y, bb.upper_boundary.y);
                real z = bb.lower_boundary.z - eps;
                real delta_z = (bb.upper_boundary.z - bb.lower_boundary.z) / resolution.z;
                int k = 0;
                bool inside = false;
                while (k < resolution.z) {
                    int kk = k;
                    Ray ray(Vector3(x, y, z), Vector3(0, 0, 1));
                    scene_geometry->query(ray);
                    if (ray.dist == Ray::DIST_INFINITE) kk = resolution.z;
                    else {
                        z += ray.dist + eps;
                        kk = std::min(int((z - bb.lower_boundary.z) / delta_z), resolution.z);
                    }
                    while (k < kk) {
                        arr.set(i, j, k, char(inside));
                        ++k;
                    }
                    inside = !inside;
                }
            }
    }

    bool inside(const Vector3 &coord) const {
        return
            0 < coord.x && coord.x < 1.f &&
            0 < coord.y && coord.y < 1.f &&
            0 < coord.z && coord.z < 1.f;
    }

    virtual Vector4 sample(const Vector3 &coord) const override {
        if (inside(coord))
            return Vector4(real(arr.sample_relative_coord(coord)));
        else
            return Vector4(0);
    }
};

TC_IMPLEMENTATION(Texture, MeshTexture, "mesh")

TC_NAMESPACE_END

