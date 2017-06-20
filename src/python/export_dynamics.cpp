/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>
#include <taichi/dynamics/fluid2d/fluid.h>
#include <taichi/dynamics/mpm2d/mpm.h>
#include <taichi/dynamics/mpm2d/mpm_particle.h>
#include <taichi/dynamics/simulation3d.h>
#include <taichi/common/asset_manager.h>

PYBIND11_MAKE_OPAQUE(std::vector<taichi::RenderParticle>);

TC_NAMESPACE_BEGIN

void export_dynamics(py::module &m) {
    m.def("register_levelset3d", &AssetManager::insert_asset<LevelSet3D>);

    py::class_<Fluid::Particle>(m, "FluidParticle")
            .def(py::init<Vector2, Vector2>())
            .def_readwrite("position", &Fluid::Particle::position)
            .def_readwrite("velocity", &Fluid::Particle::velocity)
            .def_readwrite("color", &Fluid::Particle::color)
            .def_readwrite("temperature", &Fluid::Particle::temperature);
    py::class_ <MPMParticle, std::shared_ptr<MPMParticle>> mpm_particle(m, "MPMParticle");
    mpm_particle
            .def_readwrite("position", &MPMParticle::pos)
            .def_readwrite("velocity", &MPMParticle::v)
            .def_readwrite("color", &MPMParticle::color);
    py::class_<EPParticle, std::shared_ptr<EPParticle>>(m, "EPParticle", mpm_particle)
            .def(py::init<>())
            .def_readwrite("theta_c", &EPParticle::theta_c)
            .def_readwrite("theta_s", &EPParticle::theta_s)
            .def_readwrite("mu_0", &EPParticle::mu_0)
            .def_readwrite("lambda_0", &EPParticle::lambda_0)
            .def_readwrite("hardening", &EPParticle::hardening)
            .def_readwrite("mass", &EPParticle::mass)
            .def("set_compression", &EPParticle::set_compression);
    py::class_<DPParticle, std::shared_ptr<DPParticle>>(m, "DPParticle", mpm_particle)
            .def(py::init<>())
            .def_readwrite("h_0", &DPParticle::h_0)
            .def_readwrite("h_1", &DPParticle::h_1)
            .def_readwrite("h_2", &DPParticle::h_2)
            .def_readwrite("h_3", &DPParticle::h_3)
            .def_readwrite("mu_0", &DPParticle::mu_0)
            .def_readwrite("lambda_0", &DPParticle::lambda_0)
            .def_readwrite("alpha", &DPParticle::alpha)
            .def_readwrite("q", &DPParticle::q)
            .def_readwrite("mass", &DPParticle::mass)
            .def_readwrite("phi_f", &DPParticle::phi_f);

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

    py::class_<Simulation3D, std::shared_ptr<Simulation3D>>(m, "Simulation3D")
            .def(py::init<>())
            .def("initialize", &Simulation3D::initialize)
            .def("add_particles", &Simulation3D::add_particles)
            .def("update", &Simulation3D::update)
            .def("step", &Simulation3D::step)
            .def("get_current_time", &Simulation3D::get_current_time)
            .def("get_render_particles", &Simulation3D::get_render_particles)
            .def("set_levelset", &Simulation3D::set_levelset)
            .def("get_mpi_world_rank", &Simulation3D::get_mpi_world_rank)
            .def("test", &Simulation3D::test);

    py::class_<MPM>(m, "MPMSimulator")
            .def(py::init<>())
            .def("initialize", &MPM::initialize)
            .def("step", &MPM::step)
            .def("test", &MPM::test)
            .def("add_particle", static_cast<void (MPM::*)(std::shared_ptr<MPMParticle>)>(&MPM::add_particle))
            .def("get_current_time", &MPM::get_current_time)
            .def("get_particles", &MPM::get_particles)
            .def("set_levelset", &MPM::set_levelset)
            .def("get_material_levelset", &MPM::get_material_levelset)
            .def("get_debug_blocks", &MPM::get_debug_blocks)
            .def("get_grid_block_size", &MPM::get_grid_block_size)
            .def("add_ep_particle", static_cast<void (MPM::*)(EPParticle)>(&MPM::add_particle))
            .def("add_dp_particle", static_cast<void (MPM::*)(DPParticle)>(&MPM::add_particle));

    DEFINE_VECTOR_OF_NAMED(std::shared_ptr<MPMParticle>, "MPMParticles");

    typedef std::vector<Fluid::Particle> FluidParticles;
    py::class_<FluidParticles>(m, "FluidParticles");
}

TC_NAMESPACE_END
