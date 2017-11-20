#pragma once

#include <taichi/geometry/factory.h>
#include <memory>
#include <vector>
#include <memory.h>
#include <string>
#include <functional>
#include <mutex>
#include <taichi/math/angular.h>
#include <taichi/geometry/mesh.h>
#include <taichi/visual/scene.h>
#include <taichi/math.h>

TC_NAMESPACE_BEGIN

template <int dim>
struct RigidBody {
  using Vector = VectorND<dim, real>;
  using Vectori = VectorND<dim, int>;

  using Matrix = MatrixND<dim, real>;
  using MatrixP = MatrixND<dim + 1, real>;

  using InertiaType = std::conditional_t<dim == 2, real, Matrix>;

  real mass;
  InertiaType inertia;
  Vector position, velocity, tmp_velocity;

  Vector3 color;

  // Segment mesh for 2D and thin shell for 3D
  bool codimensional;
  real friction, restitution;
  MatrixP mesh_to_local;

  AngularVelocity<dim> angular_velocity, tmp_angular_velocity;
  Rotation<dim> rotation;

  using PositionFunctionType =
      std::conditional_t<dim == 2, Function12, Function13>;
  using RotationFunctionType =
      std::conditional_t<dim == 2, Function11, Function13>;
  PositionFunctionType pos_func;
  RotationFunctionType rot_func;

  int coin;
  int id;

  Vector rotation_axis;

  using ElementType =
      std::conditional_t<dim == 2, typename ElementMesh<dim>::Elem, Triangle>;
  using MeshType = std::conditional_t<dim == 2, ElementMesh<dim>, Mesh>;

  std::shared_ptr<MeshType> mesh;
  std::mutex mut;

  RigidBody() {
    static int counter = 0;
    id = counter++;
    codimensional = false;
    mass = 0.0f;
    inertia = 0.0f;
    position = Vector(0.0f);
    velocity = Vector(0.0f);
    rotation = 0.0f;
    angular_velocity = AngularVelocity<dim>(0.0_f);
    friction = 0;
    restitution = 0;
    color = Vector3(0.5_f);
    coin = 1;
  }

  void apply_impulse(Vector impulse, Vector orig) {
    velocity += impulse / mass;
    auto torque = cross(orig - position, impulse);
    angular_velocity += get_inversed_inertia() * torque;
  }

  InertiaType get_inversed_inertia() const {
    InertiaType ret;
    TC_STATIC_IF(dim == 2) {
      ret = 1.0_f / inertia;
    }
    TC_STATIC_ELSE {
      // "Rotate" the inertia_tensor
      Matrix rotation_matrix = rotation.get_rotation_matrix();
      ret = rotation_matrix * inversed(inertia) * transposed(rotation_matrix);
    };
    TC_STATIC_END_IF
    return ret;
  }

  void apply_tmp_impulse(Vector impulse, Vector orig) {
    mut.lock();

    tmp_velocity += impulse / mass;
    auto torque = cross(orig - position, impulse);
    tmp_angular_velocity += get_inversed_inertia() * torque;

    mut.unlock();
  }

  void apply_position_impulse(Vector impulse, Vector orig) {
    position += impulse / mass;
    auto torque = cross(orig - position, impulse);
    rotation.apply_angular_velocity(get_inversed_inertia() * torque, 1);
  }

  void advance(real t, real dt) {
    if (pos_func) {
      position = pos_func(t);
      real d = 1e-3_f;
      velocity = (pos_func(t + d) - pos_func(t - d)) / (2.0_f * d);
      real vel_mag = length(velocity);
      if (velocity.abs_max() > 100.0_f) {
        TC_WARN(
            "Position not differentiable at time {}. (Magnitude {} at d = {})",
            t, vel_mag, d);
        velocity = Vector(0);
      }
    } else {
      position += velocity * dt;
    }
    if (rot_func) {
      TC_STATIC_IF(dim == 3) {
        auto rot_quat = [&](real t) {
          Vector r = radians(this->rot_func(t));
          Eigen::AngleAxis<real> angleAxisX(r[0],
                                            Eigen::Matrix<real, 3, 1>::UnitX());
          Eigen::AngleAxis<real> angleAxisY(r[1],
                                            Eigen::Matrix<real, 3, 1>::UnitY());
          Eigen::AngleAxis<real> angleAxisZ(r[2],
                                            Eigen::Matrix<real, 3, 1>::UnitZ());
          Eigen::Quaternion<real> quat(angleAxisZ * angleAxisY * angleAxisX);
          return quat;
        };
        rotation.value = rot_quat(t);
        real d = 1e-3_f;
        Eigen::AngleAxis<real> angle_axis =
            Eigen::AngleAxis<real>(rot_quat(t + d) * rot_quat(t - d).inverse());
        angle_axis.angle() *= (0.5_f / d);
        real len_rot = abs(angle_axis.angle());
        if (len_rot > 100.0_f) {
          TC_WARN(
              "Rotation not differentiable at time {}. (Magnitude {} at d = "
              "{})",
              t, len_rot, d);
          angular_velocity.value = Vector(0);
        } else {
          for (int i = 0; i < dim; i++) {
            angular_velocity.value[i] =
                angle_axis.angle() * angle_axis.axis()(i);
          }
        }
      }
      TC_STATIC_ELSE {
        auto rot_quat = [&](real t) { return radians(this->rot_func(t)); };
        rotation.value = rot_quat(t);
        real d = 1e-3_f;
        real rot = 0.5_f * (rot_quat(t + d) - rot_quat(t - d)) / d;
        real len_rot = abs(rot);
        if (len_rot > 100.0_f) {
          TC_WARN(
              "Rotation not differentiable at time {}. (Magnitude {} at d = "
              "{})",
              t, len_rot, d);
          angular_velocity.value = 0;
        } else {
          angular_velocity.value = rot;
        }
      }
      TC_STATIC_END_IF
    } else {
      rotation.apply_angular_velocity(angular_velocity, dt);
    }
  }

  Vector get_velocity_from_offset(const Vector &offset) const {
    return angular_velocity.cross(offset) + velocity;
  }

  Vector get_velocity_at(const Vector &p) const {
    return get_velocity_from_offset(p - position);
  }

  Vector get_momemtum() const {
    return mass * velocity;
  }

  void reset_tmp_velocity() {
    tmp_angular_velocity = AngularVelocity<dim>(0.0f);
    tmp_velocity = Vector(0.0f);
  }

  void apply_tmp_velocity() {
    angular_velocity += tmp_angular_velocity;
    velocity += tmp_velocity;
  }

  // Self-centered angular_momemtum
  real get_angular_momemtum() const {
    return inertia * angular_velocity;
  }

  // Returns center of mass
  Vector initialize_mass_and_inertia(real density);

  MatrixP get_world_to_local() const {
    return inversed(get_local_to_world());
  }

  MatrixP get_local_to_world() const {
    MatrixP trans(rotation.get_rotation_matrix());
    trans[dim][dim] = 1;
    return matrix_translate(&trans, position);
  }

  MatrixP get_mesh_to_world() const {
    return get_local_to_world() * mesh_to_local;
  };

  void enforce_velocity_parallel_to(Vector direction) {
    direction = normalized(direction);
    velocity = dot(velocity, direction) * direction;
  }

  void enforce_velocity_perpendicular_to(Vector direction) {
    direction = normalized(direction);
    velocity = velocity - dot(velocity, direction) * direction;
  }

  void enforce_angular_velocity_parallel_to(Vector direction) {
    direction = normalized(direction);
    TC_STATIC_IF(dim == 3) {
      angular_velocity.value =
          dot(angular_velocity.value, direction) * direction;
    }
    TC_STATIC_END_IF
  }

  InertiaType get_inv_inertia() const {
    return inversed(inertia);
  }

  // Inputs: impulse point minus position, and normal
  real get_impulse_contribution(Vector r, Vector n) const {
    real ret;
    TC_STATIC_IF(dim == 2) {
      ret = 1.0_f / mass + length2(cross(r, n)) / inertia;
    }
    TC_STATIC_ELSE {
      ret = 1.0_f / mass;
      InertiaType inversed_inertia = this->get_inversed_inertia();
      auto rn = cross_product_matrix(r);
      inversed_inertia = transposed(rn) * inversed_inertia * rn;
      ret += dot(n, inversed_inertia * n);
    }
    TC_STATIC_END_IF
    return ret;
  }
};

TC_NAMESPACE_END
