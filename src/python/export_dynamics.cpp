/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>
#include <taichi/dynamics/fluid2d/fluid.h>
#include <taichi/dynamics/simulation.h>
#include <taichi/common/asset_manager.h>

PYBIND11_MAKE_OPAQUE(std::vector<taichi::RenderParticle>);

TC_NAMESPACE_BEGIN

template <int DIM>
void register_simulation(py::module &m) {
  using Sim = Simulation<DIM>;
  py::class_<Sim, std::shared_ptr<Sim>>(
      m,
      (std::string("Simulation") + std::to_string(DIM) + std::string("D"))
          .c_str())
      .def(py::init<>())
      .def_readwrite("frame", &Sim::frame)
      .def("initialize", &Sim::initialize)
      .def("D", [](Sim *) { return Sim::dim; })
      .def("dim", [](Sim *) { return Sim::dim; })
      .def("add_particles", &Sim::add_particles)
      .def("update", &Sim::update)
      .def("step", &Sim::step)
      .def("visualize", &Sim::visualize)
      .def("get_current_time", &Sim::get_current_time)
      .def("get_render_particles", &Sim::get_render_particles)
      .def("set_levelset", &Sim::set_levelset)
      .def("get_mpi_world_rank", &Sim::get_mpi_world_rank)
      .def("get_vis_resolution", &Sim::get_vis_resolution)
      .def("get_debug_information", &Sim::get_debug_information)
      .def("general_action", &Sim::general_action)
      .def("test", &Sim::test);
}

void export_dynamics(py::module &m) {
  m.def("register_levelset2d", &AssetManager::insert_asset<LevelSet2D>);
  m.def("register_levelset3d", &AssetManager::insert_asset<LevelSet3D>);

  py::class_<Fluid>(m, "Fluid")
      .def(py::init<>())
      .def("initialize", &Fluid::initialize)
      .def("step", &Fluid::step)
      .def("add_particle", &Fluid::add_particle)
      .def("get_current_time", &Fluid::get_current_time)
      .def("get_particles", &Fluid::get_particles)
      .def("set_levelset", &Fluid::set_levelset)
      .def("get_liquid_levelset", &Fluid::get_liquid_levelset)
      .def("get_density", &Fluid::get_density)
      .def("get_pressure", &Fluid::get_pressure)
      .def("add_source", &Fluid::add_source);

  register_simulation<2>(m);
  register_simulation<3>(m);

  typedef std::vector<Fluid::Particle> FluidParticles;
  py::class_<FluidParticles>(m, "FluidParticles");
}

TC_NAMESPACE_END
