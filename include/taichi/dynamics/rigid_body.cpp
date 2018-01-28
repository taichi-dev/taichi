/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/
#include <taichi/geometry/mesh.h>
#include "rigid_body.h"
#include "inertia.h"

TC_NAMESPACE_BEGIN

// Returns center of mass
template <int dim>
typename RigidBody<dim>::Vector RigidBody<dim>::initialize_mass_and_inertia(
    real density) {
  Vector center_of_mass(0.0f);
  real volume = 0.0f;
  InertiaType inertia(0);
  TC_STATIC_IF(dim == 2) {
    const std::vector<Element<dim>> &elements = this->mesh->elements;
    int n = (int)elements.size();
    if (this->codimensional) {
      // This shell
      for (int i = 0; i < n; i++) {
        Vector a = elements[i].v[0], b = elements[i].v[1];
        real triangle_area = length(a - b);
        volume += triangle_area;
        center_of_mass += triangle_area * (a + b) * (1.0_f / 2.0_f);
      }
      center_of_mass /= volume;
      for (int i = 0; i < n; i++) {
        Vector a = elements[i].v[0], b = elements[i].v[1];
        Vector c = center_of_mass;
        // Triangle with vertices a, b, c
        int slices = 10000;  // inaccurate and hacky inertia computation
        for (int k = 0; k < slices; k++) {
          inertia += length(a - b) / slices *
                     length2(lerp((0.5_f + k) / slices, a, b) - c);
        }
      }
    } else {
      for (int i = 0; i < n; i++) {
        Vector a = elements[i].v[0], b = elements[i].v[1];
        // Triangle with vertices (0, 0), a, b
        real triangle_area = 0.5f * (a.x * b.y - a.y * b.x);
        volume += triangle_area;
        center_of_mass += triangle_area * (a + b) * (1.0_f / 3.0_f);
      }
      center_of_mass /= volume;
      for (int i = 0; i < n; i++) {
        Vector a = elements[i].v[0], b = elements[i].v[1];
        Vector c = center_of_mass;
        const Vector ac = a - c;
        const Vector ba = b - a;
        // Triangle with vertices a, b, c
        inertia += (ac[0] * ba[1] - ba[0] * ac[1]) *
                   (3 * sqr(ac[0]) + 3 * sqr(ac[1]) + sqr(ba[0]) + sqr(ba[1]) +
                    3 * ac[0] * ba[0] + 3 * ac[1] * ba[1]);
      }
      inertia = inertia * (1.0_f / 12);
    }
    TC_ASSERT_INFO(id(inertia) >= 0,
                   "Rigid body inertia cannot be negative. (Make sure vertices "
                   "are counter-clockwise)");
  }
  TC_STATIC_ELSE {
    // 3D
    std::vector<Element<dim>> triangles = mesh->elements;
    int n = triangles.size();
    if (codimensional) {
      TC_TRACE("Adding a codimensional (thin shell) rigid body");
      // Thin shell case
      // Volume is actually ``area'' in this case
      for (int i = 0; i < n; ++i) {
        Vector local_center_of_mass(0.0_f);
        for (int d = 0; d < dim; ++d) {
          Vector vert = triangles[i].v[d];
          local_center_of_mass += vert;
        }
        real local_volume =
            length(cross(triangles[i].v[1] - triangles[i].v[0],
                         triangles[i].v[2] - triangles[i].v[0])) *
            0.5_f;
        volume += local_volume;
        local_center_of_mass *= (1.0 / 3);
        center_of_mass += local_center_of_mass * local_volume;
      }
      center_of_mass /= volume;

      id(inertia) = Matrix(0.0_f);
      for (int i = 0; i < n; i++) {
        // TODO: make it more accurate
        Vector verts[dim];
        for (int d = 0; d < dim; ++d) {
          verts[d] = triangles[i].v[d] - center_of_mass;
        }
        Vector local_center_of_mass =
            (verts[0] + verts[1] + verts[2]) * (1.0_f / 3);
        Vector r = local_center_of_mass - center_of_mass;
        real local_volume =
            length(cross(triangles[i].v[1] - triangles[i].v[0],
                         triangles[i].v[2] - triangles[i].v[0])) *
            0.5_f;

        id(inertia) +=
            (Matrix(dot(r, r)) - Matrix::outer_product(r, r)) * local_volume;
      }
    } else {
      TC_TRACE("Adding a solid rigid body");
      // Solid case
      for (int i = 0; i < n; ++i) {
        Vector local_center_of_mass(0.0_f);
        Matrix vol_mat;
        for (int d = 0; d < dim; ++d) {
          Vector vert = triangles[i].v[d];
          // TC_P(vert);
          local_center_of_mass += vert;
          vol_mat[d] = vert;
        }
        real local_volume = determinant(vol_mat) / 6;
        volume += local_volume;
        local_center_of_mass *= (1.0 / 4);
        center_of_mass += local_center_of_mass * local_volume;
      }
      center_of_mass /= volume;

      inertia = id(Matrix(0.0_f));
      for (int i = 0; i < n; i++) {
        Vector verts[dim];
        for (int d = 0; d < dim; ++d) {
          verts[d] = triangles[i].v[d] - center_of_mass;
        }
        inertia += -id(tetrahedron_inertia_tensor(id(Vector(0.0_f)), verts[0],
                                                  verts[1], verts[2]));
      }
    }
  }
  TC_STATIC_END_IF

  set_inertia(inertia * density);
  set_mass(volume * density);

  TC_P(this->mass);
  TC_P(this->inertia);
  TC_P(this->inv_mass);
  TC_P(this->inv_inertia);
  TC_ASSERT_INFO(
      this->mass > 0,
      fmt::format(
          "Rigid body mass ({}) cannot be negative. (Make sure vertices "
          "are counter-clockwise)",
          this->mass));
  return center_of_mass;
}

template typename RigidBody<2>::Vector
RigidBody<2>::initialize_mass_and_inertia(real density);

template typename RigidBody<3>::Vector
RigidBody<3>::initialize_mass_and_inertia(real density);

TC_NAMESPACE_END