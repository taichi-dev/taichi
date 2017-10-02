/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/texture.h>
#include <taichi/common/asset_manager.h>

TC_NAMESPACE_BEGIN

class ZoomingTexture : public Texture {
 protected:
  std::shared_ptr<Texture> tex;
  Vector3 center;
  Vector3 inv_zoom;
  bool repeat;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
    center = config.get("center", Vector3(0.0_f));
    inv_zoom = Vector3(1.0_f) / config.get<Vector3>("zoom");
    repeat = config.get<bool>("repeat");
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    Vector3 c = inv_zoom * (coord - center) + center;
    if (repeat)
      c = fract(c);
    return tex->sample(c);
  }
};

TC_IMPLEMENTATION(Texture, ZoomingTexture, "zoom");

class LinearOpTexture : public Texture {
 protected:
  real alpha, beta;
  bool need_clamp;
  std::shared_ptr<Texture> tex1, tex2;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    alpha = config.get<real>("alpha");
    beta = config.get<real>("beta");
    need_clamp = config.get("need_clamp", false);
    tex1 = AssetManager::get_asset<Texture>(config.get<int>("tex1"));
    tex2 = AssetManager::get_asset<Texture>(config.get<int>("tex2"));
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    auto p = alpha * tex1->sample(coord) + beta * tex2->sample(coord);
    if (need_clamp) {
      for (int i = 0; i < 3; i++) {
        p[i] = clamp(p[i], 0.0_f, 1.0_f);
      }
    }
    return p;
  }
};

TC_IMPLEMENTATION(Texture, LinearOpTexture, "linear_op");

class MultiplicationTexture : public Texture {
 protected:
  std::shared_ptr<Texture> tex1;
  std::shared_ptr<Texture> tex2;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    tex1 = AssetManager::get_asset<Texture>(config.get<int>("tex1"));
    tex2 = AssetManager::get_asset<Texture>(config.get<int>("tex2"));
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    return tex1->sample(coord) * tex2->sample(coord);
  }
};

TC_IMPLEMENTATION(Texture, MultiplicationTexture, "mul");

class FractTexture : public Texture {
 protected:
  std::shared_ptr<Texture> tex;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    return fract(tex->sample(coord));
  }
};

TC_IMPLEMENTATION(Texture, FractTexture, "fract");

class RepeatedTexture : public Texture {
 protected:
  real repeat_u, repeat_v, repeat_w;
  real inv_repeat_u, inv_repeat_v, inv_repeat_w;
  std::shared_ptr<Texture> tex;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    repeat_u = config.get<real>("repeat_u");
    repeat_v = config.get<real>("repeat_v");
    repeat_w = config.get("repeat_w", 1.0_f);
    inv_repeat_u = 1.0_f / repeat_u;
    inv_repeat_v = 1.0_f / repeat_v;
    inv_repeat_w = 1.0_f / repeat_w;
    tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    real u = coord.x - floor(coord.x * repeat_u) * inv_repeat_u;
    real v = coord.y - floor(coord.y * repeat_v) * inv_repeat_v;
    real w = coord.z - floor(coord.z * repeat_w) * inv_repeat_w;
    return tex->sample(Vector3(u * repeat_u, v * repeat_v, w * repeat_w));
  }
};

TC_IMPLEMENTATION(Texture, RepeatedTexture, "repeat");

class RotatedTexture : public Texture {
 protected:
  std::shared_ptr<Texture> tex;
  int rotate_times;
  int rotate_axis;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
    rotate_times = config.get<int>("rotate_times");
    rotate_axis = config.get<int>("rotate_axis");
  }

  virtual Vector4 sample(const Vector3 &coord_) const override {
    auto coord = coord_;
    coord = coord * 2.f - Vector3(1.f, 1.f, 1.f);
    for (int i = 0; i < rotate_times; i++) {
      switch (rotate_axis) {
        case 0:
          coord = Vector3(coord.x, -coord.z, coord.y);
          break;
        case 1:
          coord = Vector3(coord.z, coord.y, -coord.x);
          break;
        case 2:
          coord = Vector3(-coord.y, coord.x, coord.z);
          break;
      }
    }
    coord = (coord + Vector3(1.f, 1.f, 1.f)) / 2.f;
    return tex->sample(coord);
  }
};

TC_IMPLEMENTATION(Texture, RotatedTexture, "rotate");

class FlippedTexture : public Texture {
 protected:
  std::shared_ptr<Texture> tex;
  int flip_axis;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
    flip_axis = config.get<int>("flip_axis");
  }

  virtual Vector4 sample(const Vector3 &coord_) const override {
    auto coord = coord_;
    coord[flip_axis] = 1.0_f - coord[flip_axis];
    return tex->sample(coord);
  }
};

TC_IMPLEMENTATION(Texture, FlippedTexture, "flip");

class BoundedTexture : public Texture {
 protected:
  std::shared_ptr<Texture> tex;
  int bound_axis;
  Vector2 bounds;
  Vector4 outside_val;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
    bound_axis = config.get<int>("axis");
    bounds = config.get<Vector2>("bounds");
    outside_val = config.get<Vector4>("outside_val");
  }

  virtual Vector4 sample(const Vector3 &coord_) const override {
    auto coord = coord_;
    if (bounds[0] <= coord[bound_axis] && coord[bound_axis] < bounds[1])
      return tex->sample(coord);
    else
      return outside_val;
  }
};

TC_IMPLEMENTATION(Texture, BoundedTexture, "bound");

class RasterizedTexture : public Texture {
 protected:
  Array2D<Vector4> cache;
  int resolution_x;
  int resolution_y;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    auto tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
    resolution_x = config.get<int>("resolution_x");
    resolution_y = config.get<int>("resolution_y");
    cache = Array2D<Vector4>(resolution_x, resolution_y);
    for (int i = 0; i < resolution_x; i++) {
      for (int j = 0; j < resolution_y; j++) {
        cache.set(i, j,
                  tex->sample(Vector2((i + 0.5f) / resolution_x,
                                      (j + 0.5f) / resolution_y)));
      }
    }
  }

  virtual Vector4 sample(const Vector2 &coord) const override {
    return cache.sample_relative_coord(coord);
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    return cache.sample_relative_coord(Vector2(coord.x, coord.y));
  }
};

TC_IMPLEMENTATION(Texture, RasterizedTexture, "rasterize");

class TranslatedTexture : public Texture {
 protected:
  std::shared_ptr<Texture> tex;
  Vector3 translation;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
    translation = config.get<Vector3>("translation");
  }

  virtual Vector4 sample(const Vector3 &coord_) const override {
    return tex->sample(coord_ - translation);
  }
};

TC_IMPLEMENTATION(Texture, TranslatedTexture, "trans");

TC_NAMESPACE_END
