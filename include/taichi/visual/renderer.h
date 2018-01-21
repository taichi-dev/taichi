/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/visual/camera.h>
#include <taichi/visual/scene.h>
#include <taichi/visual/scene_geometry.h>
#include <taichi/visualization/image_buffer.h>
#include <taichi/system/timer.h>
#include <taichi/common/interface.h>

TC_NAMESPACE_BEGIN
class Renderer : public Unit {
 public:
  virtual void initialize(const Config &config) override;
  virtual void render_stage(){};
  virtual void set_scene(std::shared_ptr<Scene> scene);
  virtual Array2D<Vector3> get_output() {
    return Array2D<Vector3>(Vector2i(width, height));
  };
  virtual void write_output(std::string fn);

 protected:
  std::shared_ptr<Camera> camera;
  std::shared_ptr<Scene> scene;
  std::shared_ptr<RayIntersection> ray_intersection;
  std::shared_ptr<SceneGeometry> sg;
  int width, height;
  int min_path_length, max_path_length;
  int num_threads;
  bool path_length_in_range(int path_length) {
    return min_path_length <= path_length && path_length <= max_path_length;
  }
};

TC_INTERFACE(Renderer);

TC_NAMESPACE_END
