/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <functional>
#include <pybind11/pybind11.h>
#include <taichi/visual/camera.h>
#include <taichi/dynamics/fluid2d/fluid.h>
#include <taichi/dynamics/simulation3d.h>
#include <taichi/visual/texture.h>
#include <taichi/visual/envmap.h>
#include <taichi/image/tone_mapper.h>
#include <taichi/io/image_reader.h>
#include <taichi/math/sdf.h>
#include <taichi/visual/renderer.h>
#include <taichi/visual/sampler.h>
#include <taichi/dynamics/poisson_solver2d.h>
#include <taichi/dynamics/poisson_solver3d.h>
#include <taichi/visual/surface_material.h>
#include <taichi/visual/volume_material.h>
#include <taichi/visual/ray_intersection.h>
#include <taichi/visual/framebuffer.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/system/benchmark.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(Camera, "camera");
TC_INTERFACE_DEF(Simulation3D, "simulation3d")
TC_INTERFACE_DEF(Texture, "texture")
TC_INTERFACE_DEF(EnvironmentMap, "envmap")
TC_INTERFACE_DEF(ToneMapper, "tone_mapper")
TC_INTERFACE_DEF(ImageReader, "image_reader")
TC_INTERFACE_DEF(SDF, "sdf")
TC_INTERFACE_DEF(Renderer, "renderer")
TC_INTERFACE_DEF(Sampler, "sampler")
TC_INTERFACE_DEF(PoissonSolver2D, "pressure_solver_2d")
TC_INTERFACE_DEF(PoissonSolver3D, "pressure_solver_3d")
TC_INTERFACE_DEF(VolumeMaterial, "volume_material")
TC_INTERFACE_DEF(SurfaceMaterial, "surface_material")
TC_INTERFACE_DEF(Framebuffer, "framebuffer")
TC_INTERFACE_DEF(Fluid, "fluid")
TC_INTERFACE_DEF(RayIntersection, "ray_intersection")
TC_INTERFACE_DEF(ParticleRenderer, "particle_renderer")
TC_INTERFACE_DEF(Benchmark, "benchmark")

TC_NAMESPACE_END
