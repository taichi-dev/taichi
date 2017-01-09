#include "flip_fluid.h"
#include <taichi/point_cloud/point_cloud.h>

TC_NAMESPACE_BEGIN

void FLIPFluid::clamp_particle(Particle & p) {
    p.position = EulerFluid::clamp_particle_position(p.position); // Avoid out of bound levelset query
    Vector2 sample_position = p.position;
    float phi = boundary_levelset.sample(sample_position) - padding;
    if (phi < 0) {
        auto grad = boundary_levelset.get_normalized_gradient(sample_position);
        p.position -= phi * grad;
        p.velocity -= dot(grad, p.velocity) * grad;
    }
}

void FLIPFluid::initialize_solver(const Config &config)
{
    EulerFluid::initialize_solver(config);
    FLIP_alpha = config.get("flip_alpha", 0.97f);
    padding = config.get("padding", 0.001f);
    advection_order = config.get("advection_order", 2);
    correction_strength = config.get("correction_strength", 0.1f);
    correction_neighbours = config.get("correction_neighbours", 5);
    u_backup = Array(width + 1, height, 0.0f, Vector2(0.0f, 0.5f));
    v_backup = Array(width, height + 1, 0.0f, Vector2(0.5f, 0.0f));
    u_count = Array(width + 1, height, 0.0f);
    v_count = Array(width, height + 1, 0.0f);
}

Vector2 FLIPFluid::sample_velocity(Vector2 position, Vector2 velocity, float lerp) {
    return EulerFluid::sample_velocity(position, u, v) +
        lerp * (velocity - EulerFluid::sample_velocity(position, u_backup, v_backup));
}

void FLIPFluid::advect(float delta_t) {
    float lerp = powf(FLIP_alpha, delta_t / 0.01f);
    float max_movement = 0.0f;
    for (auto &p : particles) {
        if (advection_order == 3) {
            Vector2 velocity_1 = sample_velocity(p.position, p.velocity, lerp);
            Vector2 velocity_2 = sample_velocity((p.position + delta_t * 0.5f * velocity_1),
                p.velocity, lerp);
            Vector2 velocity_3 = sample_velocity((p.position + delta_t * 0.75f * velocity_2),
                p.velocity, lerp);
            p.velocity =
                (2.0f / 9.0f) * velocity_1 + (3.0f / 9.0f) * velocity_2 +
                (4.0f / 9.0f) * velocity_3;
        }
        else if (advection_order == 2) {
            Vector2 velocity_1 = sample_velocity(p.position, p.velocity, lerp);
            Vector2 velocity_2 = sample_velocity(p.position - delta_t * velocity_1, p.velocity, lerp);
            p.velocity = 0.5f * (velocity_1 + velocity_2);
        }
        else if (advection_order == 1) {
            p.velocity = sample_velocity(p.position, p.velocity, lerp);
        }
        else {
            error("advection_order must be in [1, 2, 3].")
        }
        p.move(delta_t * p.velocity);
        max_movement = std::max(max_movement, length(p.velocity * delta_t));
        clamp_particle(p);
    }
}

void FLIPFluid::apply_external_forces(float delta_t) {
    for (auto &p : particles) {
        p.velocity += delta_t * gravity;
    }
}

void FLIPFluid::rasterize() {
    rasterize_component<Particle::get_velocity<0>>(u, u_count);
    rasterize_component<Particle::get_velocity<1>>(v, v_count);
}

void FLIPFluid::step(float delta_t)
{
    EulerFluid::step(delta_t);
    correct_particle_positions(delta_t);
}

void FLIPFluid::backup_velocity_field() {
    u_backup = u;
    v_backup = v;
}

void FLIPFluid::substep(float delta_t) {
    apply_external_forces(delta_t);
    mark_cells();
    rasterize();
    backup_velocity_field();
    apply_boundary_condition();
    compute_liquid_levelset();
    simple_extrapolate();
    project(delta_t);
    simple_extrapolate();
    advect(delta_t);
    t += delta_t;
}


void FLIPFluid::show(Array2D<Vector3> &buffer) {
    /*
    buffer.write_text(title, 20, 0, -1);
    if (show_grid) {
        float max_speed = get_max_grid_speed() * 2 + 1e-3f;
        auto region = Region2D(0, width + 1, 0, height + 1);
        for (auto &ind : region) {
            int i = ind.i, j = ind.j;
            buffer.set_pixel((float(i) / width), (float(j) / height), Vector3(0, 0, 0.9));
            if (v.inside(ind)) {
                buffer.set_pixel((float(i + 0.5f) / width), (float(j) / height), Vector3(0.3, 0, 0));
                for (float k = 0; k < 1; k += 0.02f) {
                    buffer.set_pixel((float(i + 0.5f) / width), (float(j + k * v[ind] / max_speed) / height), Vector3(0.5));
                }
            }
            if (u.inside(ind)) {
                buffer.set_pixel((float(i) / width), (float(j + 0.5f) / height), Vector3(0, 0.3, 0));
                for (float k = 0; k < 1; k += 0.02f) {
                    buffer.set_pixel((float(i + k * u[ind] / max_speed) / width), (float(j + 0.5f) / height), Vector3(0.5));
                }

            }
            //for (float k = 0; k < 1; k += 0.01f) {
            //    buffer.set_pixel((float(i + 0.5f) / width) + , (float(j) / height), Vector3(0.3, 0, 0));
            //}
        }
    }
    for (auto &particle : particles) {
        if (!particle.show)
            continue;
        float x = particle.position.x / width;
        float y = particle.position.y / height;
        buffer.set_pixel(x, y, Vector3(1));
    }
    */
}

FLIPFluid::FLIPFluid() : EulerFluid() {

}

void FLIPFluid::reseed() {
}

void FLIPFluid::correct_particle_positions(float delta_t, bool clear_c)
{
    if (correction_strength == 0.0f && !clear_c) {
        return;
    }
    NearestNeighbour2D nn;
    float range = 0.5f;
    std::vector<Vector2> positions;
    for (auto &p : particles) {
        positions.push_back(p.position);
    }
    nn.initialize(positions);
    std::vector<Vector2> delta_pos(particles.size());
    for (int i = 0; i < (int)particles.size(); i++) {
        delta_pos[i] = Vector2(0);
        auto &p = particles[i];
        std::vector<int> neighbour_index;
        std::vector<float> neighbour_dist;
        nn.query_n(p.position, correction_neighbours, neighbour_index, neighbour_dist);
        for (auto nei_index : neighbour_index) {
            if (nei_index == -1) {
                break;
            }
            auto &nei = particles[nei_index];
            float dist = length(p.position - nei.position);
            Vector2 dir = (p.position - nei.position) / dist;
            if (dist > 1e-4f && dist < range) {
                float a = correction_strength * delta_t * pow(1 - dist / range, 2);
                delta_pos[i] += a * dir;
                delta_pos[nei_index] -= a * dir;
            }
        }
        if (clear_c && (neighbour_index.size() <= 1 || neighbour_dist[1] > 1.5f)) {
            p.c[0] = p.c[1] = Vector2(0.0f);
        }
    }
    for (int i = 0; i < (int)particles.size(); i++) {
        particles[i].position += delta_pos[i];
        // if (delta_pos[i] != Vector2(0)) P(delta_pos[i]);
        clamp_particle(particles[i]);
    }
}

template<float(*T)(const Fluid::Particle &, const Vector2 &)>
void FLIPFluid::rasterize_component(Array & val, Array & count)
{
    val = 0;
    count = 0;
    float inv_kernel_size = 1.0f / kernel_size;
    int extent = (kernel_size + 1) / 2;
    for (auto &p : particles) {
        for (auto &ind : val.get_rasterization_region(p.position, extent)) {
            Vector2 delta_pos = ind.get_pos() - p.position;
            float weight = kernel(inv_kernel_size * delta_pos);
            val[ind] += weight * T(p, delta_pos);
            count[ind] += weight;
        }
    }
    for (auto ind : val.get_region()) {
        if (count[ind] > 0) {
            val[ind] /= count[ind];
        }
    }
}


template void FLIPFluid::rasterize_component<Fluid::Particle::get_velocity<0>>(Array &val, Array &count);
template void FLIPFluid::rasterize_component<Fluid::Particle::get_velocity<1>>(Array &val, Array &count);
template void FLIPFluid::rasterize_component<Fluid::Particle::get_affine_velocity<0>>(Array &val, Array &count);
template void FLIPFluid::rasterize_component<Fluid::Particle::get_affine_velocity<1>>(Array &val, Array &count);

TC_NAMESPACE_END
