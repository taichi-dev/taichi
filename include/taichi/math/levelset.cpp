/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "levelset.h"

TC_NAMESPACE_BEGIN

template <int DIM>
void LevelSet<DIM>::add_sphere(LevelSet<DIM>::Vector center,
                               real radius,
                               bool inside_out) {
  for (auto &ind : this->get_region()) {
    Vector sample = ind.get_pos();
    real dist = (inside_out ? -1 : 1) * (length(center - sample) - radius);
    this->set(ind, std::min(Array::get(ind), dist));
  }
}

template <>
void LevelSet<2>::add_polygon(std::vector<Vector2> polygon, bool inside_out) {
  for (auto &ind : this->get_region()) {
    Vector2 p = ind.get_pos();
    real dist = ((inside_polygon(p, polygon) ^ inside_out) ? -1 : 1) *
                (nearest_distance(p, polygon));
    this->set(ind, std::min(Array::get(ind), dist));
  }
}

template <>
Vector2 LevelSet<2>::get_gradient(const Vector2 &pos) const {
  assert_info(inside(pos),
              "LevelSet Gradient Query out of Bound! (" +
                  std::to_string(pos.x) + ", " + std::to_string(pos.y) + ")");
  real x = pos.x, y = pos.y;
  x = clamp(x - storage_offset.x, 0.0_f, this->res[0] - 1.0_f - eps);
  y = clamp(y - storage_offset.y, 0.0_f, this->res[1] - 1.0_f - eps);
  const int x_i = clamp(int(x), 0, this->res[0] - 2);
  const int y_i = clamp(int(y), 0, this->res[1] - 2);
  const real x_r = x - x_i;
  const real y_r = y - y_i;
  const real gx = lerp(y_r, Array::get(x_i + 1, y_i) - Array::get(x_i, y_i),
                       Array::get(x_i + 1, y_i + 1) - Array::get(x_i, y_i + 1));
  const real gy = lerp(x_r, Array::get(x_i, y_i + 1) - Array::get(x_i, y_i),
                       Array::get(x_i + 1, y_i + 1) - Array::get(x_i + 1, y_i));
  return Vector2(gx, gy);
}

template <int DIM>
typename LevelSet<DIM>::Vector LevelSet<DIM>::get_normalized_gradient(
    const LevelSet<DIM>::Vector &pos) const {
  Vector gradient = get_gradient(pos);
  if (length(gradient) < 1e-10f)
    gradient[0] = 1.0_f;
  return normalize(gradient);
}

template <>
real LevelSet<2>::get(const Vector2 &pos) const {
  assert_info(inside(pos),
              "LevelSet Query out of Bound! (" + std::to_string(pos.x) + ", " +
                  std::to_string(pos.y) + ")");
  real x = pos.x, y = pos.y;
  x = clamp(x - this->storage_offset.x, 0.0_f, this->res[0] - 1.0_f - eps);
  y = clamp(y - this->storage_offset.y, 0.0_f, this->res[1] - 1.0_f - eps);
  const int x_i = clamp(int(x), 0, this->res[0] - 2);
  const int y_i = clamp(int(y), 0, this->res[1] - 2);
  const real x_r = x - x_i;
  const real y_r = y - y_i;
  const real ly0 = lerp(x_r, Array::get(x_i, y_i), Array::get(x_i + 1, y_i));
  const real ly1 =
      lerp(x_r, Array::get(x_i, y_i + 1), Array::get(x_i + 1, y_i + 1));
  return lerp(y_r, ly0, ly1);
}

template <>
real LevelSet<3>::get(const Vector3 &pos) const {
  assert_info(inside(pos),
              "LevelSet Query out of Bound! (" + std::to_string(pos.x) + ", " +
                  std::to_string(pos.y) + ", " + std::to_string(pos.z) + ")");
  real x = pos.x, y = pos.y, z = pos.z;
  x = clamp(x - storage_offset.x, 0.0_f, this->res[0] - 1.0_f - eps);
  y = clamp(y - storage_offset.y, 0.0_f, this->res[1] - 1.0_f - eps);
  z = clamp(z - storage_offset.z, 0.0_f, this->res[2] - 1.0_f - eps);
  const int x_i = clamp(int(x), 0, this->res[0] - 2);
  const int y_i = clamp(int(y), 0, this->res[1] - 2);
  const int z_i = clamp(int(z), 0, this->res[2] - 2);
  const real x_r = x - x_i;
  const real y_r = y - y_i;
  const real z_r = z - z_i;
  return lerp(x_r, lerp(y_r, lerp(z_r, Array3D<real>::get(x_i, y_i, z_i),
                                  Array3D<real>::get(x_i, y_i, z_i + 1)),
                        lerp(z_r, Array3D<real>::get(x_i, y_i + 1, z_i),
                             Array3D<real>::get(x_i, y_i + 1, z_i + 1))),
              lerp(y_r, lerp(z_r, Array3D<real>::get(x_i + 1, y_i, z_i),
                             Array3D<real>::get(x_i + 1, y_i, z_i + 1)),
                   lerp(z_r, Array3D<real>::get(x_i + 1, y_i + 1, z_i),
                        Array3D<real>::get(x_i + 1, y_i + 1, z_i + 1))));
}

template <>
typename LevelSet<2>::Array LevelSet<2>::rasterize(
    LevelSet<2>::VectorI output_res) {
  for (auto &p : (*this)) {
    if (std::isnan(p)) {
      printf("Warning: nan in levelset.");
    }
  }
  Array2D<real> out(output_res);
  Vector2 actual_size;
  if (this->storage_offset == Vector2(0.0_f, 0.0_f)) {
    actual_size = Vector2(this->res[0] - 1, this->res[1] - 1);
  } else {
    actual_size = Vector2(this->res[0], this->res[1]);
  }

  Vector2 scale_factor = actual_size / output_res.template cast<real>();

  for (auto &ind :
       Region2D(0, this->res[0], 0, this->res[1], Vector2(0.5f, 0.5f))) {
    Vector2 p = scale_factor * ind.get_pos();
    out[ind] = this->sample(p);
    if (std::isnan(out[ind])) {
      out[ind] = std::numeric_limits<real>::infinity();
    }
  }
  return out;
}

template <>
Array3D<real> LevelSet<3>::rasterize(Vector3i output_res) {
  for (auto &p : (*this)) {
    if (std::isnan(p)) {
      printf("Warning: nan in levelset.");
    }
  }
  Array3D<real> out(output_res);
  Vector3 actual_size;
  if (storage_offset == Vector3(0.0_f, 0.0_f, 0.0_f)) {
    actual_size = Vector3(this->res[0] - 1, this->res[1] - 1, this->res[2] - 1);
  } else {
    actual_size = Vector3(this->res[0], this->res[1], this->res[2]);
  }

  Vector3 scale_factor = actual_size / output_res.cast<real>();

  for (auto &ind :
       Region3D(0, res[0], 0, res[1], 0, res[2], Vector3(0.5f, 0.5f, 0.5f))) {
    Vector3 p = scale_factor * ind.get_pos();
    out[ind] = sample(p);
    if (std::isnan(out[ind])) {
      out[ind] = std::numeric_limits<real>::infinity();
    }
  }
  return out;
}

template <int DIM>
void LevelSet<DIM>::add_plane(const LevelSet<DIM>::Vector &normal_, real d) {
  Vector normal = normalized(normal_);
  real coeff = 1.0_f / length(normal);
  for (auto &ind : this->get_region()) {
    Vector sample = ind.get_pos();
    real dist = (dot(sample, normal) + d) * coeff;
    this->set(ind, std::min(Array::get(ind), dist));
  }
}

template <>
void LevelSet<3>::add_cuboid(Vector3 lower_boundry,
                             Vector3 upper_boundry,
                             bool inside_out) {
  for (auto &ind : this->get_region()) {
    Vector3 sample = ind.get_pos();
    bool in_cuboid = true;
    for (int i = 0; i < 3; ++i) {
      if (!(lower_boundry[i] <= sample[i] && sample[i] <= upper_boundry[i]))
        in_cuboid = false;
    }
    real dist = INF;
    if (in_cuboid) {
      for (int i = 0; i < 3; ++i) {
        dist = std::min(dist, std::min(upper_boundry[i] - sample[i],
                                       sample[i] - lower_boundry[i]));
      }
    } else {
      Vector3 nearest_p;
      for (int i = 0; i < 3; ++i) {
        nearest_p[i] = clamp(sample[i], lower_boundry[i], upper_boundry[i]);
      }
      dist = -length(nearest_p - sample);
    }
    set(ind, inside_out ? dist : -dist);
  }
}

template <int DIM>
void LevelSet<DIM>::global_increase(real delta) {
  for (auto &ind : this->get_region()) {
    this->set(ind, Array::get(ind) + delta);
  }
}

template <>
Vector3 LevelSet<3>::get_gradient(const Vector3 &pos) const {
  assert_info(inside(pos),
              "LevelSet Gradient Query out of Bound! (" +
                  std::to_string(pos.x) + ", " + std::to_string(pos.y) + ", " +
                  std::to_string(pos.z) + ")");
  real x = pos.x, y = pos.y, z = pos.z;
  x = clamp(x - storage_offset.x, 0.0_f, res[0] - 1.0_f - eps);
  y = clamp(y - storage_offset.y, 0.0_f, res[1] - 1.0_f - eps);
  z = clamp(z - storage_offset.z, 0.0_f, res[2] - 1.0_f - eps);
  const int x_i = clamp(int(x), 0, res[0] - 2);
  const int y_i = clamp(int(y), 0, res[1] - 2);
  const int z_i = clamp(int(z), 0, res[2] - 2);
  const real x_r = x - x_i;
  const real y_r = y - y_i;
  const real z_r = z - z_i;
  // TODO: speed this up
  const real gx = lerp(y_r, lerp(z_r,
                                 Array3D<real>::get(x_i + 1, y_i, z_i) -
                                     Array3D<real>::get(x_i, y_i, z_i),
                                 Array3D<real>::get(x_i + 1, y_i, z_i + 1) -
                                     Array3D<real>::get(x_i, y_i, z_i + 1)),
                       lerp(z_r,
                            Array3D<real>::get(x_i + 1, y_i + 1, z_i) -
                                Array3D<real>::get(x_i, y_i + 1, z_i),
                            Array3D<real>::get(x_i + 1, y_i + 1, z_i + 1) -
                                Array3D<real>::get(x_i, y_i + 1, z_i + 1)));
  const real gy = lerp(z_r, lerp(x_r,
                                 Array3D<real>::get(x_i, y_i + 1, z_i) -
                                     Array3D<real>::get(x_i, y_i, z_i),
                                 Array3D<real>::get(x_i + 1, y_i + 1, z_i) -
                                     Array3D<real>::get(x_i + 1, y_i, z_i)),
                       lerp(x_r,
                            Array3D<real>::get(x_i, y_i + 1, z_i + 1) -
                                Array3D<real>::get(x_i, y_i, z_i + 1),
                            Array3D<real>::get(x_i + 1, y_i + 1, z_i + 1) -
                                Array3D<real>::get(x_i + 1, y_i, z_i + 1)));
  const real gz = lerp(x_r, lerp(y_r,
                                 Array3D<real>::get(x_i, y_i, z_i + 1) -
                                     Array3D<real>::get(x_i, y_i, z_i),
                                 Array3D<real>::get(x_i, y_i + 1, z_i + 1) -
                                     Array3D<real>::get(x_i, y_i + 1, z_i)),
                       lerp(y_r,
                            Array3D<real>::get(x_i + 1, y_i, z_i + 1) -
                                Array3D<real>::get(x_i + 1, y_i, z_i),
                            Array3D<real>::get(x_i + 1, y_i + 1, z_i + 1) -
                                Array3D<real>::get(x_i + 1, y_i + 1, z_i)));
  return Vector3(gx, gy, gz);
}

template class LevelSet<2>;

template class LevelSet<3>;

template <int DIM>
void DynamicLevelSet<DIM>::initialize(real _t0,
                                      real _t1,
                                      const LevelSet<DIM> &_ls0,
                                      const LevelSet<DIM> &_ls1) {
  t0 = _t0;
  t1 = _t1;
  levelset0 = std::make_shared<LevelSet<DIM>>(_ls0);
  levelset1 = std::make_shared<LevelSet<DIM>>(_ls1);
}

template <>
DynamicLevelSet<2>::Vector DynamicLevelSet<2>::get_spatial_gradient(
    const DynamicLevelSet<2>::Vector &pos,
    real t) const {
  Vector2 gxy0 = levelset0->get_gradient(pos);
  Vector2 gxy1 = levelset1->get_gradient(pos);
  real gx = lerp((t - t0) / (t1 - t0), gxy0.x, gxy1.x);
  real gy = lerp((t - t0) / (t1 - t0), gxy0.y, gxy1.y);
  Vector2 gradient = Vector2(gx, gy);
  if (length(gradient) < 1e-10f)
    return Vector2(1, 0);
  else
    return normalize(gradient);
}

template <>
DynamicLevelSet<3>::Vector DynamicLevelSet<3>::get_spatial_gradient(
    const DynamicLevelSet<3>::Vector &pos,
    real t) const {
  Vector3 gxyz0 = levelset0->get_gradient(pos);
  Vector3 gxyz1 = levelset1->get_gradient(pos);
  real gx = lerp((t - t0) / (t1 - t0), gxyz0.x, gxyz1.x);
  real gy = lerp((t - t0) / (t1 - t0), gxyz0.y, gxyz1.y);
  real gz = lerp((t - t0) / (t1 - t0), gxyz0.z, gxyz1.z);
  Vector3 gradient = Vector3(gx, gy, gz);
  if (length(gradient) < 1e-10f)
    return Vector3(1, 0, 0);
  else
    return normalize(gradient);
}

template <int DIM>
real DynamicLevelSet<DIM>::get_temporal_derivative(
    const typename DynamicLevelSet<DIM>::Vector &pos,
    real t) const {
  real l0 = levelset0->get(pos);
  real l1 = levelset1->get(pos);
  return (l1 - l0) / (t1 - t0);
}

template <int DIM>
real DynamicLevelSet<DIM>::sample(
    const typename DynamicLevelSet<DIM>::Vector &pos,
    real t) const {
  real l1 = levelset0->get(pos);
  real l2 = levelset1->get(pos);
  return lerp((t - t0) / (t1 - t0), l1, l2);
}

template <>
Array3D<real> DynamicLevelSet<3>::rasterize(Vector3i res, real t) {
  Array3D<real> r0 = levelset0->rasterize(res);
  Array3D<real> r1 = levelset1->rasterize(res);
  Array3D<real> out(res);
  for (auto &ind : Region3D(Vector3i(0), res, Vector3(0.5f, 0.5f, 0.5f))) {
    out[ind] = lerp((t - t0) / (t1 - t0), r0[ind], r1[ind]);
    if (std::isnan(out[ind])) {
      out[ind] = std::numeric_limits<real>::infinity();
    }
  }
  return out;
}

template <>
Array2D<real> DynamicLevelSet<2>::rasterize(Vector2i res, real t) {
  Array2D<real> r0 = levelset0->rasterize(res);
  Array2D<real> r1 = levelset1->rasterize(res);
  Array2D<real> out(res);
  for (auto &ind : Region2D(Vector2i(0), res, Vector2(0.5f, 0.5f))) {
    out[ind] = lerp((t - t0) / (t1 - t0), r0[ind], r1[ind]);
    if (std::isnan(out[ind])) {
      out[ind] = std::numeric_limits<real>::infinity();
    }
  }
  return out;
}

template class DynamicLevelSet<2>;

template class DynamicLevelSet<3>;

TC_NAMESPACE_END
