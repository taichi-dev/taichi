/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/visual/texture.h>
#include <taichi/visual/scene_geometry.h>
#include <taichi/visualization/image_buffer.h>
#include <taichi/math/array_3d.h>
#include <taichi/math/levelset.h>
#include <taichi/math/math.h>
#include <taichi/common/asset_manager.h>
#include <taichi/geometry/mesh.h>

TC_NAMESPACE_BEGIN

class ConstantTexture final : public Texture {
 private:
  Vector4 val;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    val = config.get<Vector4>("value");
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
    image.load_image(config.get<std::string>("filename"));
  }

  bool inside(const Vector3 &coord) const {
    return 0 <= coord.x && coord.x <= 1.0 && 0 <= coord.y && coord.y <= 1.0 &&
           0 <= coord.z && coord.z <= 1.0;
  }

  virtual Vector4 sample(const Vector3 &coord_) const override {
    Vector2 coord(coord_.x - floor(coord_.x), coord_.y - floor(coord_.y));
    if (inside(coord_))
      return image.sample_relative_coord(coord);
    else
      return Vector4(0);
  }
};

TC_IMPLEMENTATION(Texture, ImageTexture, "image");

class TextTexture : public Texture {
 protected:
  Array2D<Vector4> image;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    int width = config.get<int>("width");
    int height = config.get<int>("height");
    std::string font_file_fn = config.get<std::string>("font_file");
    std::string content = config.get<std::string>("content");
    real size = config.get<real>("size");
    int dx = config.get<int>("dx");
    int dy = config.get<int>("dy");
    image.initialize(Vector2i(width, height));
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
    bounds = config.get<Vector3>("bounds") * 0.5_f;
  }

  bool inside(const Vector3 &coord) const {
    Vector3 c = coord - Vector3(0.5f);
    return std::abs(c.x) < bounds.x && std::abs(c.y) < bounds.y &&
           std::abs(c.z) < bounds.z;
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    return Vector4(inside(coord) ? 1.0_f : 0.0_f);
  }
};

TC_IMPLEMENTATION(Texture, RectTexture, "rect");

class RingTexture : public Texture {
 protected:
  real inner, outer;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    inner = config.get("inner", 0.0_f) / 2.0f;
    outer = config.get("outer", 1.0_f) / 2.0f;
  }

  static bool inside(Vector2 p, Vector2 c, real r) {
    return (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) <= r * r;
  }

  virtual Vector4 sample(const Vector2 &coord) const override {
    return Vector4(inside(coord, Vector2(0.5f, 0.5f), outer) &&
                           !inside(coord, Vector2(0.5f, 0.5f), inner)
                       ? 1.0_f
                       : 0.0_f);
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
    scale = config.get("scale", 1.0_f);
    real rotation = config.get("rotation", 0.0_f);
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
    Vector2 p(p_.x * rotation_c + p_.y * rotation_s,
              -p_.x * rotation_s + p_.y * rotation_c);
    p += Vector2(0.5f);
    if (!inside(p, Vector2(0.50f, 0.50f), 0.5f)) {
      return true;
    }
    if (!inside(p, Vector2(0.50f, 0.50f), 0.5f * scale)) {
      return false;
    }
    p = Vector2(0.5f) + (p - Vector2(0.5f)) * (1.0_f / scale);
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
    return Vector4(is_white(coord) ? 1.0_f : 0.0_f);
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
    tex1 = AssetManager::get_asset<Texture>(config.get<int>("tex1"));
    tex2 = AssetManager::get_asset<Texture>(config.get<int>("tex2"));
    repeat_u = config.get<real>("repeat_u");
    repeat_v = config.get<real>("repeat_v");
    repeat_w = config.get("repeat_w", 1.0_f);
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    int p = (int)floor(coord.x * repeat_u), q = (int)floor(coord.y * repeat_v),
        r = (int)floor(coord.z * repeat_w);
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
    coeff_u = config.get<real>("coeff_u");
    coeff_v = config.get<real>("coeff_v");
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
    center = config.get<Vector3>("center");
    radius = config.get<real>("radius");
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    return Vector4(real(bool(length(coord - center) < radius)));
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
    steps = config.get<Vector4>("steps");
    base = config.get<Vector4>("base");
    increment = config.get<Vector4>("increment");
  }

  virtual Vector4 sample(const Vector3 &coord_) const override {
    Vector4 coord(coord_, 0.0_f);
    return base + (coord * steps).floor() * increment;
  }
};

TC_IMPLEMENTATION(Texture, SlicedTexture, "sliced");

class FastMeshTexture : public Texture {
 protected:
  Array3D<real> arr;
  Vector3i resolution;

 public:
  // parameter name for mesh path is 'filename'
  void initialize(const Config &config) override {
    Texture::initialize(config);
    Mesh mesh;
    mesh.initialize(config);
    Scene scene;
    Vector3 scale = config.get("scale", Vector3(1));
    Vector3 translate = config.get("translate", Vector3(0));
    bool adaptive = config.get("adaptive", true);
    auto mesh_p = std::make_shared<Mesh>(mesh);
    Matrix4 trans(1);
    trans = matrix4_scale(&trans, scale);
    trans = matrix4_translate(&trans, translate);
    mesh_p->transform = trans;
    scene.add_mesh(mesh_p);
    scene.finalize_geometry();
    auto ray_intersection = create_instance<RayIntersection>("embree");
    std::shared_ptr<SceneGeometry> scene_geometry(
        new SceneGeometry(std::make_shared<Scene>(scene), ray_intersection));
    BoundingBox bb = mesh.get_bounding_box();
    // calc offset and delta_x to keep the mesh in the center of texture
    resolution = config.get<Vector3i>("resolution");
    arr = Array3D<real>(resolution);
    real delta_x = 0;
    Vector3i offset;
    for (int i = 0; i < 3; ++i)
      delta_x = std::max(
          delta_x,
          (bb.upper_boundary[i] - bb.lower_boundary[i]) / resolution[i]);
    for (int i = 0; i < 3; ++i)
      offset[i] = int(resolution[i] -
                      (bb.upper_boundary[i] - bb.lower_boundary[i]) / delta_x) /
                  2;
    if (!adaptive) {
      offset = Vector3i(0);
      delta_x = 1.0_f / resolution[0];
      bb.lower_boundary = Vector3(0);
    }
    // cast ray for each i, j
    for (int i = 0; i < resolution.x; ++i) {
      for (int j = 0; j < resolution.y; ++j) {
        real x = (i - offset.x) * delta_x + bb.lower_boundary.x;
        real y = (j - offset.y) * delta_x + bb.lower_boundary.y;
        real z = (0 - offset.z) * delta_x + bb.lower_boundary.z;
        real base_z = z - eps;
        int k = 0;
        bool inside = false;
        while (k < resolution.z) {
          int next_k;
          Ray ray(Vector3(x, y, z), Vector3(0, 0, 1));
          scene_geometry->query(ray);
          if (ray.dist == Ray::DIST_INFINITE) {
            next_k = resolution.z;
            inside = false;
          } else {
            z += ray.dist;
            next_k = std::min(int((z - base_z) / delta_x), resolution.z);
          }
          while (k < next_k) {
            arr.set(i, j, k, inside ? 1 : 0);
            ++k;
          }
          inside = !inside;
        }
      }
    }
  }

  bool inside(const Vector3 &coord) const {
    return 0 < coord.x && coord.x < 1.f && 0 < coord.y && coord.y < 1.f &&
           0 < coord.z && coord.z < 1.f;
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    if (inside(coord)) {
      return Vector4(real(arr.sample_relative_coord(coord)));
    } else
      return Vector4(0);
  }
};

TC_IMPLEMENTATION(Texture, FastMeshTexture, "fast_mesh")

class MeshTexture : public Texture {
 protected:
  std::shared_ptr<SceneGeometry> scene_geometry;
  int mesh_accuracy = 3;

 public:
  // parameter name for mesh path is 'filename'
  void initialize(const Config &config) override {
    Texture::initialize(config);
    mesh_accuracy = config.get("mesh_accuracy", mesh_accuracy);
    Mesh mesh;
    mesh.initialize(config);
    Scene scene;
    Vector3 scale = config.get("scale", Vector3(1));
    Vector3 translate = config.get("translate", Vector3(0));
    auto mesh_p = std::make_shared<Mesh>(mesh);
    Matrix4 trans(1);
    trans = matrix4_scale(&trans, scale);
    if (config.get("adaptive", true)) {
      BoundingBox bb = mesh.get_bounding_box();
      translate -= (bb.lower_boundary + bb.upper_boundary) / 2.0_f * scale;
    }
    trans = matrix4_translate(&trans, translate);
    mesh_p->transform = trans;
    scene.add_mesh(mesh_p);
    scene.finalize_geometry();
    auto ray_intersection = create_instance<RayIntersection>("embree");
    scene_geometry = std::make_shared<SceneGeometry>(
        std::make_shared<Scene>(scene), ray_intersection);

    if (config.get("generate_peeling", false)) {
      real peeling_cycle = config.get("peeling_cycle", 1.0_f);
      real peeling_velocity = config.get("peeling_velocity", 1.0_f);
      std::ofstream os;
      std::string fn =
          std::getenv("TAICHI_ROOT_DIR") +
          std::string("/taichi/projects/mpm/data/apple_peeler.pos");
      os.open(fn, std::ios::app);
      Vector3 pos;
      real t_start = -1.0_f;
      bool last_intersection = false;
      for (real t = 0;; t += 0.001_f) {
        pos.y = t * peeling_velocity;
        real angle;
        if (t_start < 0.0_f) {
          angle = 0.0_f;
        } else {
          angle = 2.0_f * pi / peeling_cycle * (t - t_start);
        }
        pos.x = 0.5_f + sin(angle);
        pos.z = 0.5_f + cos(angle);
        Vector3 dir = normalize(Vector3(0.5_f, pos.y, 0.5_f) - pos);
        Ray ray(pos, dir);
        scene_geometry->query(ray);
        if (ray.dist == Ray::DIST_INFINITE || ray.dist > 1.0_f) {
          if (last_intersection)
            break;
          last_intersection = false;
        } else {
          if (!last_intersection)
            t_start = t;
          pos += ray.dist * dir;
          os << t - t_start << " " << pos.x << " " << pos.y << " " << pos.z
             << std::endl;
          if (t - t_start > 10.0_f)
            break;
          last_intersection = true;
        }
      }
      os.close();
      TC_STOP;
    }
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    auto emitting_ray_test = [&]() {
      int inside = 0;
      real alpha = rand() * 2.0_f * pi;
      real beta = rand() * 2.0_f * pi;
      Vector3 direction(
          Vector3(cos(alpha) * cos(beta), cos(alpha) * sin(beta), sin(alpha)));
      Vector3 position = coord;
      while (true) {
        Ray ray(position, direction);
        scene_geometry->query(ray);
        if (ray.dist == Ray::DIST_INFINITE) {
          break;
        } else {
          position += ray.dist * direction;
          inside = 1 - inside;
        }
      }
      return inside;
    };
    int test_count[2];
    test_count[0] = test_count[1] = 0;
    for (int i = 0; i < mesh_accuracy; ++i) {
      int test_result = emitting_ray_test();
      test_count[test_result]++;
    }
    return Vector4(test_count[1] > test_count[0] ? 1.0_f : 0.0_f);
  }
};

TC_IMPLEMENTATION(Texture, MeshTexture, "mesh")

class LevelSet3DTexture : public Texture {
 protected:
  std::shared_ptr<LevelSet3D> levelset;
  Vector2 bounds;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    levelset = AssetManager::get_asset<LevelSet3D>(config.get<int>("levelset"));
    bounds = config.get<Vector2>("bounds");
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    real value = levelset->sample_relative_coord(coord);
    return (bounds.x < value && value < bounds.y) ? Vector4(1) : Vector4(0);
  }
};

TC_IMPLEMENTATION(Texture, LevelSet3DTexture, "levelset3d")

class PolygonTexture : public Texture {
  // TODO: test it
 protected:
  ElementMesh<2> poly;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    poly.initialize(config);
  }

  virtual Vector4 sample(const Vector3 &pos_) const override {
    Vector2 pos(pos_);
    real min_distance = 1e30_f;
    for (auto &elem : poly.elements) {
      // clamp to ends, signed
      real dist = distance_to_segment(pos, elem.v[0], elem.v[1], true, true);
      if (abs(dist) < abs(min_distance)) {
        min_distance = dist;
      }
    }
    return Vector4(min_distance);
  }
};

TC_IMPLEMENTATION(Texture, PolygonTexture, "polygon")

TC_NAMESPACE_END
