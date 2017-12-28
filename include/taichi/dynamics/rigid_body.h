#pragma once

#include <taichi/geometry/factory.h>
#include <taichi/visual/scene_geometry.h>
#include <memory>
#include <vector>
#include <memory.h>
#include <string>
#include <functional>
#include <mutex>
#include <taichi/math/angular.h>
#include <taichi/geometry/mesh.h>
#include <taichi/math.h>

TC_NAMESPACE_BEGIN

template <int dim>
struct RigidBody {
 public:
  using Vector = VectorND<dim, real>;
  using Vectori = VectorND<dim, int>;

  using Matrix = MatrixND<dim, real>;
  using MatrixP = MatrixND<dim + 1, real>;

  using InertiaType = std::conditional_t<dim == 2, real, Matrix>;
  using PositionFunctionType =
      std::conditional_t<dim == 2, Function12, Function13>;
  using RotationFunctionType =
      std::conditional_t<dim == 2, Function11, Function13>;

  using ElementType = typename ElementMesh<dim>::Elem;
  using MeshType = ElementMesh<dim>;

  // Segment mesh for 2D and thin shell for 3D
  bool codimensional;
  real frictions[2], restitution;
  MatrixP mesh_to_centroid;
  real mass, inv_mass;
  InertiaType inertia, inv_inertia;
  Vector position, velocity, tmp_velocity;
  real linear_damping, angular_damping;
  Vector3 color;
  AngularVelocity<dim> angular_velocity, tmp_angular_velocity;
  Rotation<dim> rotation;

  int id;
  Vector rotation_axis;

  std::unique_ptr<MeshType> mesh;

  // Will not be serialized:
  std::unique_ptr<std::mutex> mut;
  int pos_func_id = -1, rot_func_id = -1;
  PositionFunctionType pos_func;
  RotationFunctionType rot_func;

  TC_IO_DECL {
    TC_IO(codimensional, frictions, restitution, mesh_to_centroid, mass,
          inv_mass);
    TC_IO(inertia, inv_inertia);
    TC_IO(position, velocity, tmp_velocity);
    TC_IO(linear_damping, angular_damping);
    TC_IO(color);
    TC_IO(angular_velocity, tmp_angular_velocity);
    TC_IO(rotation);
    TC_IO(id, rotation_axis);
    TC_IO(mesh);
    TC_IO(pos_func_id, rot_func_id);
  }

  RigidBody() {
    static int counter = 0;
    id = counter++;
    codimensional = false;
    mass = 0.0f;
    inertia = 0.0f;
    position = Vector(0.0f);
    velocity = Vector(0.0f);
    rotation = Rotation<dim>();
    angular_velocity = AngularVelocity<dim>();
    frictions[0] = 0;
    frictions[1] = 0;
    restitution = 0;
    linear_damping = 0;
    angular_damping = 0;
    color = Vector3(0.5_f);
    mesh_to_centroid = MatrixP::identidy();
    mut = std::make_unique<std::mutex>();
  }

  void set_as_background() {
    set_infinity_inertia();
    set_infinity_mass();
    pos_func = [](real t) { return Vector(0); };
    TC_STATIC_IF(dim == 2) {
      rot_func = [](real t) { return 0.0_f; };
    }
    TC_STATIC_ELSE {
      rot_func = [](real t) { return Vector(0.0_f); };
    }
    TC_STATIC_END_IF
  }

  void apply_impulse(Vector impulse, Vector orig) {
    velocity += impulse * get_inv_mass();
    apply_torque(cross(orig - position, impulse));
  }

  void apply_torque(const typename AngularVelocity<dim>::ValueType &torque) {
    angular_velocity += get_transformed_inversed_inertia() * torque;
  }

  InertiaType get_transformed_inertia() const {
    InertiaType ret;
    TC_STATIC_IF(dim == 2) {
      ret = inertia;
    }
    TC_STATIC_ELSE {
      // "Rotate" the inertia_tensor
      Matrix rotation_matrix = rotation.get_rotation_matrix();
      ret = rotation_matrix * inertia * transposed(rotation_matrix);
    };
    TC_STATIC_END_IF
    return ret;
  }

  InertiaType get_transformed_inversed_inertia() const {
    InertiaType ret;
    TC_STATIC_IF(dim == 2) {
      ret = this->get_inv_inertia();
    }
    TC_STATIC_ELSE {
      // "Rotate" the inertia_tensor
      Matrix rotation_matrix = rotation.get_rotation_matrix();
      ret = rotation_matrix * this->get_inv_inertia() *
            transposed(rotation_matrix);
    };
    TC_STATIC_END_IF
    return ret;
  }

  void apply_tmp_impulse(Vector impulse, Vector orig) {
    mut->lock();

    tmp_velocity += impulse * get_inv_mass();
    auto torque = cross(orig - position, impulse);
    tmp_angular_velocity += get_transformed_inversed_inertia() * torque;

    mut->unlock();
  }

  void apply_position_impulse(Vector impulse, Vector orig) {
    position += impulse * get_inv_mass();
    auto torque = cross(orig - position, impulse);
    rotation.apply_angular_velocity(get_transformed_inversed_inertia() * torque,
                                    1);
  }

  void advance(real t, real dt) {
    if (pos_func) {
      position = pos_func(t);
      real d = 1e-4_f;
      velocity = (pos_func(t + d) - pos_func(t - d)) / (2.0_f * d);
      real vel_mag = length(velocity);
      if (velocity.abs_max() > 50.0_f) {
        TC_WARN(
            "Position not differentiable at time {}. (Magnitude {} at d = {})",
            t, vel_mag, d);
        velocity = Vector(0);
      }
    } else {
      velocity *= std::exp(-linear_damping * dt);
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
        real d = 1e-4_f;
        real rot = 0.5_f * (rot_quat(t + d) - rot_quat(t - d)) / d;
        real len_rot = abs(rot);
        if (len_rot > 1000.0_f) {
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
      angular_velocity.value *= std::exp(-angular_damping * dt);
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
    tmp_angular_velocity = AngularVelocity<dim>();
    tmp_velocity = Vector(0.0f);
  }

  void apply_tmp_velocity() {
    angular_velocity += tmp_angular_velocity;
    velocity += tmp_velocity;
  }

  // Self-centered angular_momemtum
  typename AngularVelocity<dim>::ValueType get_angular_momemtum() const {
    return inertia * angular_velocity.value;
  }

  // Returns center of mass
  Vector initialize_mass_and_inertia(real density);

  MatrixP get_centroid_to_world() const {
    MatrixP trans(rotation.get_rotation_matrix());
    trans[dim][dim] = 1;
    return matrix_translate(&trans, position);
  }

  MatrixP get_mesh_to_world() const {
    return get_centroid_to_world() * mesh_to_centroid;
  }

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

  // Inputs: impulse point minus position, and normal
  real get_impulse_contribution(Vector r, Vector n) const {
    real ret;
    TC_STATIC_IF(dim == 2) {
      ret =
          this->get_inv_mass() + length2(cross(r, n)) * this->get_inv_inertia();
    }
    TC_STATIC_ELSE {
      ret = inv_mass;
      InertiaType inversed_inertia = this->get_transformed_inversed_inertia();
      auto rn = cross_product_matrix(r);
      inversed_inertia = transposed(rn) * inversed_inertia * rn;
      ret += dot(n, inversed_inertia * n);
    }
    TC_STATIC_END_IF
    return ret;
  }

  real get_mass() const {
    return mass;
  }

  real get_inv_mass() const {
    return inv_mass;
  }

private:
  // These are in local space. Only transformed versions are public.

  InertiaType get_inertia() const {
    return inertia;
  }

  InertiaType get_inv_inertia() const {
    return inv_inertia;
  }
public:

  void set_mass(real mass) {
    TC_ASSERT(std::isnormal(mass));
    this->mass = mass;
    this->inv_mass = 1.0_f / mass;
  }

  void set_infinity_mass() {
    // this->mass = std::numeric_limits<real>::infinity();
    this->mass = 1e30_f;
    this->inv_mass = 0.0_f;
  }

  void set_inertia(const InertiaType &inertia) {
    this->inertia = inertia;
    TC_STATIC_IF(dim == 2) {
      this->inv_inertia = inversed(inertia);
    }
    TC_STATIC_ELSE {
      this->inv_inertia =
          inversed(inertia.template cast<float64>()).template cast<real>();
    }
    TC_STATIC_END_IF
  }

  void set_infinity_inertia() {
    // this->inertia = InertiaType(std::numeric_limits<real>::infinity());
    this->inertia = InertiaType(1e17_f);
    this->inv_inertia = InertiaType(0.0_f);
  }
};

TC_NAMESPACE_END
