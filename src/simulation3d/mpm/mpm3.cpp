/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm3.h"

#ifdef TC_USE_MPI

#include <mpi.h>

#endif

#include <taichi/math/qr_svd.h>
#include <taichi/system/threading.h>
#include <taichi/visual/texture.h>
#include <taichi/math/math_util.h>
#include <taichi/math/math_simd.h>
#include <taichi/common/asset_manager.h>
#include <taichi/system/profiler.h>

TC_NAMESPACE_BEGIN

#define TC_MPM_USE_LOCKS
#ifdef TC_MPM_USE_LOCKS
#define LOCK_GRID grid_locks[ind].lock();
#define UNLOCK_GRID grid_locks[ind].unlock();
#else
#define LOCK_GRID
#define UNLOCK_GRID
#endif

// Note: assuming abs(x) <= 2!!
inline real w(real x) {
    x = abs(x);
#ifdef CV_ON
    assert(x <= 2);
#endif
    if (x < 1) {
        return 0.5f * x * x * x - x * x + 2.0f / 3.0f;
    } else {
        return -1.0f / 6.0f * x * x * x + x * x - 2 * x + 4.0f / 3.0f;
    }
}

// Note: assuming abs(x) <= 2!!
inline real dw(real x) {
    real s = x < 0.0f ? -1.0f : 1.0f;
    x *= s;
#ifdef CV_ON
    assert(x <= 2.0f);
#endif
    real val;
    real xx = x * x;
    if (x < 1.0f) {
        val = 1.5f * xx - 2.0f * x;
    } else {
        val = -0.5f * xx + 2.0f * x - 2.0f;
    }
    return s * val;
}

/*
inline real w(const Vector3 &a) {
    return w(a.x) * w(a.y) * w(a.z);
}

inline Vector3 dw(const Vector3 &a) {
    return Vector3(dw(a.x) * w(a.y) * w(a.z), w(a.x) * dw(a.y) * w(a.z), w(a.x) * w(a.y) * dw(a.z));
}
*/

#define PREPROCESS_KERNELS\
    Vector4s w_cache[3]; \
    Vector4s dw_cache[3];\
    Vector p_fract = fract(p.pos); \
    for (int k = 0; k < 3; k++) { \
        const Vector4s t = Vector4s(p_fract[k]) - Vector4s(-1, 0, 1, 2); \
        auto tt = t * t; \
        auto ttt = tt * t; \
        w_cache[k] = Vector4s(-1 / 6.0f, 0.5f, -0.5f, 1 / 6.0f) * ttt + \
        Vector4s(1, -1, -1, 1) * tt + \
        Vector4s(-2, 0, 0, 2) * t + \
        Vector4s(4 / 3.0f, 2 / 3.0f, 2 / 3.0f, 4 / 3.0f); \
        dw_cache[k] = Vector4s(-0.5f, 1.5f, -1.5f, 0.5f) * tt + \
        Vector4s(2, -2, -2, 2) * t + \
        Vector4s(-2, 0, 0, 2); \
    } \
    const int base_i = int(floor(p.pos[0])) - 1; \
    const int base_j = int(floor(p.pos[1])) - 1; \
    const int base_k = int(floor(p.pos[2])) - 1;

#define CALCULATE_WEIGHT \
    const real weight = w_cache[0][ind.i - base_i] * w_cache[1][ind.j - base_j] * w_cache[2][ind.k - base_k];

#define CALCULATE_GRADIENT \
    const Vector dw( \
            dw_cache[0][ind.i - base_i] * w_cache[1][ind.j - base_j] * w_cache[2][ind.k - base_k], \
            w_cache[0][ind.i - base_i] * dw_cache[1][ind.j - base_j] * w_cache[2][ind.k - base_k], \
            w_cache[0][ind.i - base_i] * w_cache[1][ind.j - base_j] * dw_cache[2][ind.k - base_k] \
    );


void MPM3D::initialize(const Config &config) {
    Simulation3D::initialize(config);
    res = config.get_vec3i("resolution");
    gravity = config.get_vec3("gravity");
    use_mpi = config.get("use_mpi", false);
    apic = config.get("apic", true);
    async = config.get("async", false);
    mpi_initialized = false;
    if (use_mpi) {
#ifndef TC_USE_MPI
        error("Not compiled with MPI. Please recompile with cmake -DTC_USE_MPI=True")
#else
        assert_info(!async, "AsyncMPM support is not finished.");
        MPI_Init(nullptr, nullptr);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
#endif
    } else {
        mpi_world_size = 1;
        mpi_world_rank = 0;
    }
    base_delta_t = config.get("base_delta_t", 1e-6f);
    cfl = config.get("cfl", 1.0f);
    strength_dt_mul = config.get("strength_dt_mul", 1.0f);
    TC_LOAD_CONFIG(affine_damping, 0.0f);
    if (async) {
        maximum_delta_t = config.get("maximum_delta_t", 1e-1f);
    } else {
        maximum_delta_t = base_delta_t;
    }

    grid_velocity.initialize(res + Vector3i(1), Vector(0.0f), Vector3(0.0f));
    grid_mass.initialize(res + Vector3i(1), 0, Vector3(0.0f));
    grid_velocity_and_mass.initialize(res + Vector3i(1), Vector4(0.0f), Vector3(0.0f));
    grid_locks.initialize(res + Vector3i(1), 0, Vector3(0.0f));
    scheduler.initialize(res, base_delta_t, cfl, strength_dt_mul, &levelset, mpi_world_rank);
}

void MPM3D::add_particles(const Config &config) {
    std::shared_ptr<Texture> density_texture = AssetManager::get_asset<Texture>(config.get_int("density_tex"));
    for (int i = 0; i < res[0]; i++) {
        for (int j = 0; j < res[1]; j++) {
            for (int k = 0; k < res[2]; k++) {
                Vector3 coord = Vector3(i + 0.5f, j + 0.5f, k + 0.5f) / Vector3(res);
                real num = density_texture->sample(coord).x;
                int t = (int)num + (rand() < num - int(num));
                for (int l = 0; l < t; l++) {
                    MPM3Particle *p = nullptr;
                    if (config.get("type", std::string("ep")) == std::string("ep")) {
                        p = new EPParticle3();
                        p->initialize(config);
                    } else {
                        p = new DPParticle3();
                        p->initialize(config);
                    }
                    p->pos = Vector(i + rand(), j + rand(), k + rand());
                    p->mass = 1.0f;
                    p->v = config.get("initial_velocity", p->v);
                    p->last_update = current_t_int;
                    particles.push_back(p);
                    scheduler.insert_particle(p, true);
                }
            }
        }
    }
    P(particles.size());
}

std::vector<RenderParticle> MPM3D::get_render_particles() const {
    using Particle = RenderParticle;
    std::vector<Particle> render_particles;
    render_particles.reserve(particles.size());
    Vector3 center(res[0] / 2.0f, res[1] / 2.0f, res[2] / 2.0f);
    for (auto p_p : particles) {
        MPM3Particle &p = *p_p;
        // at least synchronize the position
        Vector3 pos = p.pos - center + (current_t_int - p.last_update) * base_delta_t * p.v;
        if (p.state == MPM3Particle::UPDATING) {
            render_particles.push_back(Particle(pos, Vector4(0.8f, 0.1f, 0.2f, 0.5f)));
        } else if (p.state == MPM3Particle::BUFFER) {
            render_particles.push_back(Particle(pos, Vector4(0.8f, 0.8f, 0.2f, 0.5f)));
        } else {
            render_particles.push_back(Particle(pos, Vector4(0.8f, 0.9f, 1.0f, 0.5f)));
        }
    }
    return render_particles;
}

void MPM3D::resample() {
    real alpha_delta_t = 1;
    if (apic)
        alpha_delta_t = 0;
    parallel_for_each_active_particle([&](MPM3Particle &p) {
        real delta_t = base_delta_t * (current_t_int - p.last_update);
        Vector v(0.0f), bv(0.0f);
        Matrix cdg(0.0f);
        Matrix b(0.0f);
        Vector3 pos = p.pos;
        int count = 0;
        PREPROCESS_KERNELS
        for (auto &ind : get_bounded_rasterization_region(pos)) {
            count++;
            CALCULATE_WEIGHT
            CALCULATE_GRADIENT
            const Vector grid_vel = grid_velocity[ind];
            const Vector weight_grid_vel = weight * grid_vel;
            v += weight_grid_vel;
            const Vector aa = weight_grid_vel;
            const Vector bb = Vector3(ind.i, ind.j, ind.k) - pos;
            Matrix out(aa[0] * bb[0], aa[1] * bb[0], aa[2] * bb[0],
                       aa[0] * bb[1], aa[1] * bb[1], aa[2] * bb[1],
                       aa[0] * bb[2], aa[1] * bb[2], aa[2] * bb[2]);
            b += out;
#ifdef TC_MPM_WITH_FLIP
            bv += weight * grid_velocity_backup[ind];
#endif
            cdg += glm::outerProduct(grid_vel, dw);
            CV(grid_vel);
        }
        if (count != 64 || !apic) {
            b = Matrix(0);
        }
        // We should use an std::exp here, but that is too slow...
        real damping = std::max(0.0f, 1.0f - delta_t * affine_damping);
        p.apic_b = b * damping;
        cdg = Matrix(1) + delta_t * cdg;
#ifdef TC_MPM_WITH_FLIP
        // APIC part + FLIP part
        p.v = (1 - alpha_delta_t) * v + alpha_delta_t * (v - bv + p.v);
#else
        p.v = v;
#endif
        Matrix dg = cdg * p.dg_e * p.dg_p;
#ifdef CV_ON
        if (abnormal(dg) || abnormal(cdg) || abnormal(p.dg_e) || abnormal(p.dg_cache)) {
            P(dg);
            P(cdg);
            P(p.dg_e);
            P(p.dg_p);
            P(p.dg_cache);
            error("");
        }
#endif
        p.dg_e = cdg * p.dg_e;
        p.dg_cache = dg;
#ifdef CV_ON
        if (abnormal(dg) || abnormal(cdg) || abnormal(p.dg_e) || abnormal(p.dg_cache)) {
            P(dg);
            P(cdg);
            P(p.dg_e);
            P(p.dg_p);
            P(p.dg_cache);
            error("");
        }
#endif
    });
}

void MPM3D::calculate_force_and_rasterize(real delta_t) {
    {
        Profiler _("calculate force");
        parallel_for_each_active_particle([&](MPM3Particle &p) {
            p.calculate_force();
        });
    }
    TC_PROFILE("reset velocity_and_mass", grid_velocity_and_mass.reset(Vector4s(0.0f)));
    TC_PROFILE("reset velocity", grid_velocity.reset(Vector(0.0f)));
    TC_PROFILE("reset mass", grid_mass.reset(0.0f));
    {
        Profiler _("rasterize velocity, mass, and force");
        parallel_for_each_active_particle([&](MPM3Particle &p) {
            PREPROCESS_KERNELS
            const Vector pos = p.pos, v = p.v;
            const real mass = p.mass;
            const Matrix apic_b_3_mass = p.apic_b * (3.0f * mass);
            const Vector3 mass_v = mass * v;
            const Matrix delta_t_tmp_force = delta_t * p.tmp_force;
            Vector4s delta_velocity_and_mass;
            delta_velocity_and_mass[3] = mass;
            for (auto &ind : get_bounded_rasterization_region(pos)) {
                Vector3 d_pos = Vector(ind.i, ind.j, ind.k) - pos;
                CALCULATE_WEIGHT
                CALCULATE_GRADIENT
                // Originally
                // v + 3 * apic_b * d_pos;
                Vector3 rast_v = mass_v + (apic_b_3_mass * d_pos);
                delta_velocity_and_mass[0] = rast_v[0];
                delta_velocity_and_mass[1] = rast_v[1];
                delta_velocity_and_mass[2] = rast_v[2];

                const Vector force = delta_t_tmp_force * dw;
                const Vector4 delta_from_force = Vector4(force.x, force.y, force.z, 0.0f);
                CV(force);
                CV(p.tmp_force);
                CV(gw);
                LOCK_GRID
                grid_velocity_and_mass[ind] += weight * delta_velocity_and_mass + delta_from_force;
                UNLOCK_GRID
            }
        });
    }
    {
        Profiler _("normalize");
        for (auto ind : grid_mass.get_region()) {
            auto &velocity_and_mass = grid_velocity_and_mass[ind];
            const real mass = velocity_and_mass[3];
            if (mass > 0) {
                grid_mass[ind] = mass;
                CV(grid_velocity[ind]);
                CV(1 / grid_mass[ind]);
                grid_velocity[ind] = (1.0f / mass) * (*reinterpret_cast<Vector3 *>(&velocity_and_mass));
                CV(grid_velocity[ind]);
            }
        }
    }
#ifdef TC_MPM_WITH_FLIP
    error("grid_back_velocity is not in the correct position");
    grid_backup_velocity();
#endif
}

void MPM3D::grid_apply_boundary_conditions(const DynamicLevelSet3D &levelset, real t) {
    for (auto &ind : scheduler.get_active_grid_points()) {
        Vector3 pos = Vector3(0.5 + ind[0], 0.5 + ind[1], 0.5 + ind[2]);
        real phi = levelset.sample(pos, t);
        if (1 < phi || phi < -3) continue;
        Vector3 n = levelset.get_spatial_gradient(pos, t);
        Vector boundary_velocity = levelset.get_temporal_derivative(pos, t) * n;
        Vector3 v = grid_velocity[ind] - boundary_velocity;
        if (phi > 0) { // 0~1
            real pressure = std::max(-glm::dot(v, n), 0.0f);
            real mu = levelset.levelset0->friction;
            if (mu < 0) { // sticky
                v = Vector3(0.0f);
            } else {
                Vector3 t = v - n * glm::dot(v, n);
                if (length(t) > 1e-6f) {
                    t = normalize(t);
                }
                real friction = -clamp(glm::dot(t, v), -mu * pressure, mu * pressure);
                v = v + n * pressure + t * friction;
            }
        } else if (phi < 0.0f) {
            v = n * std::max(0.0f, glm::dot(v, n));
        }
        v += boundary_velocity;
        grid_velocity[ind] = v;
    }
}

void MPM3D::particle_collision_resolution(real t) {
    parallel_for_each_active_particle([&](MPM3Particle &p) {
        if (p.state == MPM3Particle::UPDATING) {
            p.resolve_collision(levelset, t);
        }
    });
}

void MPM3D::substep() {
    Profiler _p("mpm_substep");
    synchronize_particles();
    if (!particles.empty()) {
        scheduler.update_particle_groups();
        scheduler.reset_particle_states();
        old_t_int = current_t_int;
        if (async) {
            scheduler.reset();
            scheduler.update_dt_limits(current_t);

            original_t_int_increment = std::min(get_largest_pot(int64(maximum_delta_t / base_delta_t)),
                                                scheduler.update_max_dt_int(current_t_int));

            t_int_increment = original_t_int_increment - current_t_int % original_t_int_increment;

            current_t_int += t_int_increment;
            current_t = current_t_int * base_delta_t;

            scheduler.set_time(current_t_int);

            scheduler.expand(false, true);
        } else {
            // sync
            t_int_increment = 1;
            scheduler.states = 2;
            for (auto &p : particles) {
                p->state = MPM3Particle::UPDATING;
            }
            current_t_int += t_int_increment;
            current_t = current_t_int * base_delta_t;
        }
        if (!use_mpi) {
            TC_PROFILE("update", scheduler.update());
        }
        TC_PROFILE("calculate_force_and_rasterize", calculate_force_and_rasterize(t_int_increment * base_delta_t));
        TC_PROFILE("external_force", grid_apply_external_force(gravity, t_int_increment * base_delta_t));
        TC_PROFILE("boundary_condition", grid_apply_boundary_conditions(levelset, current_t));
#ifdef CV_ON
        for (auto &p: particles) {
            if (abnormal(p->dg_e)) {
                P(p->dg_e);
                error("abnormal DG_e (before resampling)");
            }
        }
#endif
        TC_PROFILE("resample", resample());
        if (!async) {
            for (auto &p: particles) {
                assert_info(p->state == MPM3Particle::UPDATING, "should be updating");
            }
#ifdef CV_ON
            for (auto &p: particles) {
                if (abnormal(p->dg_e)) {
                    P(p->dg_e);
                    error("abnormal DG_e");
                }
            }
            for (auto &p: this->scheduler.active_particles) {
                if (abnormal(p->dg_e)) {
                    P(p->dg_e);
                    error("abnormal DG_e in active_particles");
                }
            }
#endif
        }
        {
            Profiler _("plasticity");
            // TODO: should this be active particle?
            parallel_for_each_particle([&](MPM3Particle &p) {
                if (p.state == MPM3Particle::UPDATING) {
                    p.pos += (current_t_int - p.last_update) * base_delta_t * p.v;
                    p.last_update = current_t_int;
                    p.pos.x = clamp(p.pos.x, 0.0f, res[0] - eps);
                    p.pos.y = clamp(p.pos.y, 0.0f, res[1] - eps);
                    p.pos.z = clamp(p.pos.z, 0.0f, res[2] - eps);
                    p.plasticity();
                }
            });
        }
        TC_PROFILE("particle_collision", particle_collision_resolution(current_t));
        if (async) {
            scheduler.enforce_smoothness(original_t_int_increment);
        }
    }
}

#define TC_MPM_TAG_BELONGING 1
#define TC_MPM_TAG_PARTICLE_COUNT 2
#define TC_MPM_TAG_PARTICLES 3

void MPM3D::synchronize_particles() {
    if (!use_mpi) {
        // No need for this
        return;
    }
#ifdef TC_USE_MPI
    // Count number of particles with in each block
    Array3D<int> self_particle_count(scheduler.res, 0);
    P(scheduler.get_active_particles().size());
    for (auto p: scheduler.get_active_particles()) {
        self_particle_count[scheduler.get_rough_pos(p)] += 1;
    }
    auto old_belonging = scheduler.belonging;
    if (mpi_world_rank == 0) { // Master
        // Receive particle counts
        Array3D<int> particle_count_all(scheduler.res, 0);
        Array3D<int> particle_count(scheduler.res, 0);
        for (int i = 0; i < mpi_world_size; i++) {
            if (i != 0) {
                // Fetch from slaves
                MPI_Recv(static_cast<void *>(&particle_count.get_data()[0]), particle_count.get_size(), MPI_INT, i,
                         TC_MPM_TAG_PARTICLE_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int sum = 0;
                for (auto &ind: particle_count.get_region()) {
                    sum += particle_count[ind];
                    if (particle_count[ind] > 0) {
                        //P(scheduler.belonging[ind]);
                    }
                }
                //P(sum);
            } else {
                particle_count = self_particle_count;
            }
            for (auto &ind: particle_count.get_region()) {
                if (scheduler.belonging[ind] == i) {
                    particle_count_all[ind] = particle_count[ind];
                }
            }
        }
        // Re-decomposition according to x-axis slices
        std::vector<int> total_num_particles_x((size_t)scheduler.res[0], 0);
        std::vector<int> belonging_x((size_t)scheduler.res[0], 0);
        Array3D<int> belonging(scheduler.res);
        int total_num_particles = 0;
        for (auto &ind: particle_count_all.get_region()) {
            total_num_particles_x[ind.i] += particle_count_all[ind];
            total_num_particles += particle_count_all[ind];
        }
        // Determine slices
        int accumulated_num_particles = 0;
        int head = 0;
        int threshold = total_num_particles / mpi_world_size + 1;
        P(total_num_particles);
        P(threshold);
        for (int i = 0; i < scheduler.res[0]; i++) {
            accumulated_num_particles += total_num_particles_x[i];
            while (accumulated_num_particles >= threshold) {
                accumulated_num_particles -= threshold;
                head += 1;
            }
            belonging_x[i] = head;
            // P(head);
        }
        // Broadcast into y and z
        for (auto &ind: belonging.get_region()) {
            belonging[ind] = belonging_x[ind.i];
        }
        scheduler.belonging = belonging;
        for (int i = 1; i < mpi_world_size; i++) {
            // Send partition information to other nodes
            MPI_Send((void *)&scheduler.belonging.get_data()[0], scheduler.belonging.get_size(), MPI_INT, i,
                     TC_MPM_TAG_BELONGING, MPI_COMM_WORLD);
        }
    } else {
        // Slaves
        // Send particle count to master
        MPI_Send(&self_particle_count.get_data()[0], self_particle_count.get_size(), MPI_INT, 0,
                 TC_MPM_TAG_PARTICLE_COUNT,
                 MPI_COMM_WORLD);
        // Receive new domain decomposition information
        MPI_Recv((void *)&scheduler.belonging.get_data()[0], scheduler.belonging.get_size(), MPI_INT, 0,
                 TC_MPM_TAG_BELONGING,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int block_count = 0;
        for (auto ind: scheduler.belonging.get_region()) {
            block_count += scheduler.belonging[ind] == mpi_world_rank;
        }
        P(block_count);
    }
    if (!mpi_initialized) {
        // Delete particles outside
        // During first iteration, clear particles outside before synchronization
        clear_particles_outside();
    }
    std::vector<std::vector<EPParticle3>> particles_to_send;
    particles_to_send.resize(mpi_world_size);
    for (int i = 0; i < mpi_world_size; i++) {
        particles_to_send[i] = std::vector<EPParticle3>();
    }
    // Exchange (a small amount of) particles to other nodes
    for (auto &p: scheduler.get_active_particles()) {
        int belongs_to = scheduler.belongs_to(p);
        if (belongs_to != mpi_world_rank) {
            particles_to_send[belongs_to].push_back(*(EPParticle3 *)p);
        }
    }
    P(scheduler.get_active_particles().size());
    std::vector<char> to_receive(0);
    for (int i = 0; i < mpi_world_size; i++) {
        if (i == mpi_world_rank) continue;
        if (i < mpi_world_rank) {
            // Send, and then receive
            int to_send = particles_to_send[i].size();
            MPI_Send(&to_send, 1, MPI_INT, i, TC_MPM_TAG_PARTICLES, MPI_COMM_WORLD);
            if (to_send)
                MPI_Send(&particles_to_send[i][0], to_send * sizeof(EPParticle3), MPI_CHAR, i,
                         TC_MPM_TAG_PARTICLES,
                         MPI_COMM_WORLD);
            int to_recv;
            MPI_Recv(&to_recv, 1, MPI_INT, i, TC_MPM_TAG_PARTICLES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            to_receive.resize(to_recv * sizeof(EPParticle3));
            if (to_recv) {
                MPI_Recv(&to_receive[0], to_recv * sizeof(EPParticle3), MPI_CHAR, i, TC_MPM_TAG_PARTICLES,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else if (i > mpi_world_rank) {
            // Receive, and then send
            int to_recv;
            MPI_Recv(&to_recv, 1, MPI_INT, i, TC_MPM_TAG_PARTICLES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            to_receive.resize(to_recv * sizeof(EPParticle3));
            if (to_recv)
                MPI_Recv(&to_receive[0], to_recv * sizeof(EPParticle3), MPI_CHAR, i, TC_MPM_TAG_PARTICLES,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int to_send = particles_to_send[i].size();
            MPI_Send(&to_send, 1, MPI_INT, i, TC_MPM_TAG_PARTICLES, MPI_COMM_WORLD);
            if (to_send) {
                MPI_Send(&particles_to_send[i][0], particles_to_send[i].size() * sizeof(EPParticle3), MPI_CHAR, i,
                         TC_MPM_TAG_PARTICLES,
                         MPI_COMM_WORLD);
            }
        }
        for (int i = 0; i < to_receive.size() / sizeof(EPParticle3); i++) {
            EPParticle3 *ptr = new EPParticle3(*(EPParticle3 *)&to_receive[i * sizeof(EPParticle3)]);
            scheduler.get_active_particles().push_back(ptr);
        }
    }
    for (auto &p: scheduler.get_active_particles()) {
        p->state = MPM3Particle::UPDATING;
        int b = scheduler.belongs_to(p);
        p->color = Vector3(b % 2, b / 2 % 2, b / 4 % 2);
    }
    if (mpi_initialized) {
        // Delete particles outside
        clear_particles_outside();
    }
    particles = scheduler.active_particles;
    mpi_initialized = true;
#endif
}

void MPM3D::clear_particles_outside() {
    std::vector<MPM3Particle *> new_active_particles;
    for (auto p: scheduler.get_active_particles()) {
        if (scheduler.belongs_to(p) == mpi_world_rank) {
            new_active_particles.push_back(p);
        } else {
            // delete p;
        }
    }
    scheduler.get_active_particles() = new_active_particles;
}

void MPM3D::finalize() {
#ifdef TC_USE_MPI
    MPI_Finalize();
#endif
}

bool MPM3D::test() const {
    for (int i = 0; i < 100000; i++) {
        Matrix3 m(1.000000238418579101562500000000, -0.000000000000000000000000000000,
                  -0.000000000000000000000220735070, 0.000000000000000000000000000000, 1.000000238418579101562500000000,
                  -0.000000000000000000216840434497, 0.000000000000000000000211758237,
                  -0.000000000000000001084202172486, 1.000000000000000000000000000000);
        Matrix3 u, sig, v;
        svd(m, u, sig, v);
        if (!is_normal(sig)) {
            P(m);
            P(u);
            P(sig);
            P(v);
        }
    }
    return false;
}

MPM3D::~MPM3D() {
    for (auto &p : particles) {
        delete p;
    }
}

TC_IMPLEMENTATION(Simulation3D, MPM3D, "mpm");

TC_NAMESPACE_END
