/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/volume_material.h>
#include <taichi/visual/texture.h>
#include <taichi/math/array_3d.h>
#include <taichi/math/stencils.h>
#include <taichi/common/asset_manager.h>
#include <queue>

TC_NAMESPACE_BEGIN

TC_IMPLEMENTATION(VolumeMaterial, VolumeMaterial, "homogeneous");

class VacuumVolumeMaterial : public VolumeMaterial {
 public:
  virtual void initialize(const Config &config) override {
    VolumeMaterial::initialize(config);
    this->volumetric_scattering = 0.0_f;
    this->volumetric_absorption = 0.0_f;
  }

  virtual real get_attenuation(real dist) const override { return 1.0_f; }

  virtual real unbiased_sample_attenuation(const Vector3 &start,
                                           const Vector3 &end,
                                           StateSequence &rand) const override {
    return 1.0_f;
  }

  virtual VolumeEvent sample_event(StateSequence &rand,
                                   const Ray &ray) const override {
    error("invalid");
    return VolumeEvent::absorption;
  }

  virtual bool is_vacuum() const override { return true; }
};

TC_IMPLEMENTATION(VolumeMaterial, VacuumVolumeMaterial, "vacuum");

class VoxelVolumeMaterial : public VolumeMaterial {
 protected:
  Array3D<real> voxels;
  std::shared_ptr<Texture> tex;
  Vector3i resolution;
  real maximum;

 public:
  virtual void initialize(const Config &config) override {
    VolumeMaterial::initialize(config);
    this->volumetric_scattering = config.get<real>("scattering");
    this->volumetric_absorption = config.get<real>("absorption");
    this->resolution = config.get<Vector3i>("resolution");
    this->tex = AssetManager::get_asset<Texture>(config.get<int>("tex"));
    voxels.initialize(resolution, 1.0_f);
    maximum = 0.0_f;
    Vector3 inv = Vector3(1.0_f) / resolution.cast<real>();
    for (auto &ind : voxels.get_region()) {
      voxels[ind] = tex->sample(ind.get_pos() * inv).x;
      assert_info(voxels[ind] >= 0.0_f, "Density can not be negative.");
      maximum = std::max(maximum, voxels[ind]);
    }
  }

  virtual real sample_free_distance(StateSequence &rand,
                                    const Ray &ray) const override {
    int counter = 0;
    real kill;
    real dist = 0.0_f;
    real tot = volumetric_scattering + volumetric_absorption;
    do {
      counter += 1;
      if (counter > 10000) {
        printf("Warning: path too long\n");
        break;
      }
      dist += -log(1 - rand()) / (maximum * tot);
      Vector3 pos = ray.orig + ray.dir * dist;
      pos = multiply_matrix4(world2local, pos, 1.0_f);
      if (pos.x < 0 || pos.x >= 1 || pos.y < 0 || pos.y >= 1 || pos.z < 0 ||
          pos.z >= 1) {
        // Outside the texture
        dist = std::numeric_limits<real>::infinity();
        break;
      }
      kill = voxels.sample_relative_coord(pos);
    } while (maximum * rand() > kill && dist < ray.dist);
    return dist;
  }

  virtual real unbiased_sample_attenuation(const Vector3 &start,
                                           const Vector3 &end,
                                           StateSequence &rand) const override {
    auto dir = normalized(end - start);
    return sample_free_distance(rand, Ray(start, dir)) >= length(end - start);
  }
};

TC_IMPLEMENTATION(VolumeMaterial, VoxelVolumeMaterial, "voxel");

class SDFVoxelVolumeMaterial : public VoxelVolumeMaterial {
 protected:
  Array3D<real> sdf;

  // Signed distance field
  void calculate_sdf() {
    Array3D<Vector3> nearest(resolution, Vector3(1e30f));
    // Priority queue returns biggest element
    std::priority_queue<std::pair<real, Index3D>> pq;
    for (auto &ind : voxels.get_region()) {
      if (voxels[ind] > 0) {
        sdf[ind] = 0.0_f;
        nearest[ind] = ind.get_pos();
        pq.push(std::make_pair(-sdf[ind], ind));
      }
    }
    Vector3 inv_res = Vector3(1.0_f) / Vector3(resolution.cast<real>());
    // Dijkstra
    while (!pq.empty()) {
      auto t = pq.top();
      real dist = -t.first;
      Index3D ind = t.second;
      pq.pop();
      if (dist > sdf[ind]) {
        // Already outdated
        continue;
      }
      Vector3 target = nearest[ind];
      for (auto &st : neighbour6_3d) {
        Index3D nei_ind = ind + st;
        if (voxels.inside(nei_ind)) {
          real d = length(multiply_matrix4(
              local2world, (nei_ind.get_pos() - target) * inv_res, 0));
          if (d < sdf[nei_ind]) {
            // Update
            nearest[nei_ind] = target;
            sdf[nei_ind] = d;
            pq.push(std::make_pair(-d, nei_ind));
          }
        }
      }
    }
    real shrink =
        2 * std::max(std::max(length(multiply_matrix4(
                                  local2world, Vector3(1, 0, 0) * inv_res, 0)),
                              length(multiply_matrix4(
                                  local2world, Vector3(0, 1, 0) * inv_res, 0))),
                     length(multiply_matrix4(local2world,
                                             Vector3(0, 0, 1) * inv_res, 0)));
    for (auto &ind : sdf.get_region()) {
      sdf[ind] -= shrink;
    }
  }

 public:
  virtual void initialize(const Config &config) override {
    VoxelVolumeMaterial::initialize(config);
    sdf.initialize(resolution, 1e30f);
    calculate_sdf();
  }

  virtual real sample_free_distance(StateSequence &rand,
                                    const Ray &ray) const override {
    int counter = 0;
    real kill;
    real dist = 0.0_f;
    real tot = volumetric_scattering + volumetric_absorption;
    do {
      counter += 1;
      if (counter > 10000) {
        printf("Warning: path too long\n");
        break;
      }

      Vector3 pos;
      for (int i = 0; i < 100; i++) {
        pos = multiply_matrix4(world2local, ray.orig + ray.dir * dist, 1.0_f);
        if (inside_unit_cube(pos)) {
          real d = sdf.sample_relative_coord(pos);
          if (d > 1e-4_f) {
            dist += d;
            continue;
          }
        }
        break;
      }
      dist += -log(1 - rand()) / (maximum * tot);
      pos = ray.orig + ray.dir * dist;
      pos = multiply_matrix4(world2local, pos, 1.0_f);
      if (!inside_unit_cube(pos)) {
        // Outside the texture
        dist = std::numeric_limits<real>::infinity();
        break;
      }
      kill = voxels.sample_relative_coord(pos);
    } while (maximum * rand() > kill && dist < ray.dist);
    return dist;
  }
};

TC_IMPLEMENTATION(VolumeMaterial, SDFVoxelVolumeMaterial, "sdf_voxel");

VolumeStack::VolumeStack() {
  static std::shared_ptr<VolumeMaterial> vacuum =
      create_instance<VolumeMaterial>("vacuum");
  stack.push_back(vacuum.get());
}

TC_NAMESPACE_END
