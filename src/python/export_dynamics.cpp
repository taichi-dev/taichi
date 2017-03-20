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

using namespace boost::python;

EXPLICIT_GET_POINTER(taichi::MPMParticle);

EXPLICIT_GET_POINTER(taichi::EPParticle);

EXPLICIT_GET_POINTER(taichi::DPParticle);

EXPLICIT_GET_POINTER(taichi::Simulation3D);

EXPLICIT_GET_POINTER(taichi::Fluid);

TC_NAMESPACE_BEGIN

void export_dynamics() {
    def("create_fluid", create_instance<Fluid>);
    def("create_simulation3d", create_instance<Simulation3D>);
    class_<Fluid::Particle>("FluidParticle", init<Vector2, Vector2>())
        .def_readwrite("position", &Fluid::Particle::position)
        .def_readwrite("velocity", &Fluid::Particle::velocity)
        .def_readwrite("color", &Fluid::Particle::color)
        .def_readwrite("temperature", &Fluid::Particle::temperature);
    class_<MPMParticle, std::shared_ptr<MPMParticle>>("MPMParticle")
        .def_readwrite("position", &MPMParticle::pos)
        .def_readwrite("velocity", &MPMParticle::v)
        .def_readwrite("color", &MPMParticle::color);
    class_<EPParticle, std::shared_ptr<EPParticle>, bases<MPMParticle >>("EPParticle")
        .def_readwrite("theta_c", &EPParticle::theta_c)
        .def_readwrite("theta_s", &EPParticle::theta_s)
        .def_readwrite("mu_0", &EPParticle::mu_0)
        .def_readwrite("lambda_0", &EPParticle::lambda_0)
        .def_readwrite("hardening", &EPParticle::hardening)
        .def_readwrite("mass", &EPParticle::mass)
        .def("set_compression", &EPParticle::set_compression);
    class_<DPParticle, std::shared_ptr<DPParticle>, bases<MPMParticle >>("DPParticle")
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

    class_<Fluid>("Fluid")
        .def("initialize", &Fluid::initialize)
        .def("step", &Fluid::step)
        .def("add_particle", &Fluid::add_particle)
        .def("get_current_time", &Fluid::get_current_time)
        .def("get_particles", &Fluid::get_particles)
        .def("set_levelset", &Fluid::set_levelset)
        .def("get_liquid_levelset", &Fluid::get_liquid_levelset)
        .def("get_density", &Fluid::get_density)
        .def("get_pressure", &Fluid::get_pressure)
        .def("add_source", &Fluid::add_source)
        ;

#define EXPORT_SIMULATOR_3D(SIM) \
        class_<SIM>(#SIM) \
        .def("initialize", &SIM::initialize) \
        .def("update", &SIM::update) \
        .def("step", &SIM::step) \
        .def("get_current_time", &SIM::get_current_time) \
        .def("get_render_particles", &SIM::get_render_particles) \
        ;
    EXPORT_SIMULATOR_3D(Simulation3D);

#define EXPORT_MPM(SIM) \
    class_<SIM>(#SIM "Simulator") \
        .def("initialize", &SIM::initialize) \
        .def("step", &SIM::step) \
        .def("add_particle", static_cast<void (SIM::*)(std::shared_ptr<MPMParticle>)>(&SIM::add_particle)) \
        .def("get_current_time", &SIM::get_current_time) \
        .def("get_particles", &SIM::get_particles) \
        .def("set_levelset", &SIM::set_levelset) \
        .def("get_material_levelset", &SIM::get_material_levelset) \
        .def("add_ep_particle", static_cast<void (SIM::*)(EPParticle)>(&SIM::add_particle)) \
        .def("add_dp_particle", static_cast<void (SIM::*)(DPParticle)>(&SIM::add_particle)) \
        ;

    EXPORT_MPM(MPM);

    DEFINE_VECTOR_OF_NAMED(std::shared_ptr<MPMParticle>, "MPMParticles");


    typedef std::vector<Fluid::Particle> FluidParticles;
    class_<FluidParticles>("FluidParticles")
        .def(vector_indexing_suite<FluidParticles>());
    register_ptr_to_python<std::shared_ptr<Simulation3D>>();
    register_ptr_to_python<std::shared_ptr<Fluid>>();
}

TC_NAMESPACE_END
