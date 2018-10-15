/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <cmath>
#include <taichi/util.h>
#include "math.h"
#include "linalg.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Geometry>
#pragma GCC diagnostic pop
#include <type_traits>

TC_NAMESPACE_BEGIN

template <int dim>
class AngularVelocity {
 public:
  using Vector = VectorND<dim, real>;
  using ValueType = typename std::conditional_t<dim == 2, real, Vector>;

  ValueType value;

  TC_IO_DEF(value);

  AngularVelocity() {
    value = ValueType(0.0_f);
  }
  AngularVelocity(const ValueType &value) : value(value){};

  AngularVelocity &operator+=(const AngularVelocity &o) {
    *this = *this + o;
    return *this;
  }

  AngularVelocity operator+(const AngularVelocity &o) const {
    return AngularVelocity(value + o.value);
  }

  AngularVelocity operator*(real val) const {
    return AngularVelocity(value * val);
  }

  Vector cross(const Vector &input) const {
    Vector ret;
    TC_STATIC_IF(dim == 2) {
      ret = Vector(-input.y, input.x) * value;
    }
    TC_STATIC_ELSE {
      ret = taichi::cross(id(value), id(input));
    }
    TC_STATIC_END_IF;
    return ret;
  }
};

template <int dim>
class Rotation {
 public:
  using Vector = VectorND<dim, real>;
  using Matrix = MatrixND<dim, real>;
  using AngVel = AngularVelocity<dim>;
  using ValueType =
      typename std::conditional_t<dim == 2, real, Eigen::Quaternion<real>>;

  ValueType value;

  TC_IO_DEF(value);

  Rotation() {
    TC_STATIC_IF(dim == 2) {
      value = 0;
    }
    TC_STATIC_ELSE {
      value = id(Eigen::Quaternion<real>(1, 0, 0, 0));
    }
    TC_STATIC_END_IF
    return;
  }

  explicit Rotation(real value) {
    // according to dim
    TC_STATIC_IF(dim == 2) {
      id(this->value) = value;
    }
    TC_STATIC_ELSE {
      id(this->value) = Eigen::Quaternion<real>(1, 0, 0, 0);
    }
    TC_STATIC_END_IF
    return;
  }

  Rotation operator-() const {
    Rotation ret;
    TC_STATIC_IF(dim == 2) {
      ret = Rotation(-value);
    }
    TC_STATIC_ELSE{TC_NOT_IMPLEMENTED} TC_STATIC_END_IF return ret;
  }

  Matrix get_rotation_matrix() const {
    Matrix ret;
    TC_STATIC_IF(dim == 2) {
      ret[0][0] = std::cos(id(value));
      ret[1][0] = -std::sin(id(value));
      ret[0][1] = std::sin(id(value));
      ret[1][1] = std::cos(id(value));
    }
    TC_STATIC_ELSE {
      auto mat = id(value).toRotationMatrix();
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          ret[i][j] = mat(j, i);
        }
      }
    }
    TC_STATIC_END_IF
    return ret;
  }

  void apply_angular_velocity(const AngVel &vel, real dt) {
    TC_STATIC_IF(dim == 2) {
      value += id(dt * vel.value);
    }
    TC_STATIC_ELSE {
      Vector3 axis(id(vel).value[0], id(vel).value[1], id(vel).value[2]);
      real angle = length(axis);
      if (angle < 1e-10_f) {
        return;
      }
      axis = normalized(axis);
      real ot = angle * dt;
      real s = std::sin(ot / 2);
      real c = std::cos(ot / 2);
      Eigen::Quaternion<real> omega_t(c, s * axis[0], s * axis[1], s * axis[2]);
      value = id(omega_t) * value;
    }
    TC_STATIC_END_IF
    return;
  }

  Vector rotate(const Vector &vector) const {
    return get_rotation_matrix() * vector;
  }
};

inline VectorND<2, real> rotate(const VectorND<2, real> &v, real angle) {
  return Rotation<2>(angle).rotate(v);
}

inline VectorND<2, real> rotate(const VectorND<2, real> &v,
                                const Rotation<2> &rot) {
  return rot.rotate(v);
}

inline VectorND<3, real> rotate(const VectorND<3, real> &v,
                                const Rotation<3> &rot) {
  return rot.rotate(v);
}

template <typename S>
struct IO<Eigen::Quaternion<real>, S> {
  using implemented = std::true_type;

  void operator()(S &serializer, const Eigen::Quaternion<real> &t) {
    serializer("x", t.x());
    serializer("y", t.y());
    serializer("z", t.z());
    serializer("w", t.w());
  }
};

TC_NAMESPACE_END
