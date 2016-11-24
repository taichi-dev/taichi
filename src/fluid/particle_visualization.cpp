#include "particle_visualization.h"

TC_NAMESPACE_BEGIN

    void ParticleShadowMapRenderer::render(ImageBuffer<Vector3> &buffer, const std::vector<Particle> particles) {
        std::vector <std::pair<real, int>> indices(particles.size());
        for (int i = 0; i < (int) indices.size(); i++) {
            indices[i] = std::make_pair(-glm::dot(light_direction, particles[i].position), i);
        }
        std::vector<float> occlusion(particles.size());
        Array2D<float> occlusion_buffer(shadow_map_resolution, shadow_map_resolution);
        std::sort(indices.begin(), indices.end());
        Vector3 u = abs(light_direction.y) > 0.99f ? Vector3(1, 0, 0) : glm::cross(light_direction, Vector3(0, 1, 0));
        u = normalized(u);
        Vector3 v = normalized(glm::cross(u, light_direction));

        for (int i = 0; i < (int) indices.size(); i++) {
            const int index = indices[i].second;
            Vector2 coord(glm::dot(particles[index].position - center, u),
                          glm::dot(particles[index].position - center, v));
            coord = coord * 0.3f + 0.5f;
            if (0 <= coord.x && coord.x < 1 && 0 <= coord.y && coord.y < 1) {
                occlusion[index] = occlusion_buffer.sample_relative_coord(coord);
                int x = (int) round(coord.x * occlusion_buffer.get_width());
                int y = (int) round(coord.y * occlusion_buffer.get_height());
                if (occlusion_buffer.inside(x, y)) {
                    occlusion_buffer[x][y] += shadowing;
                }
            } else {
                occlusion[index] = 0.0f;
            }
        }

        float rotate_cos = cos(rotate_z);
        float rotate_sin = sin(rotate_z);
        float front_angle = 0.3f;
        Matrix3 mat = Matrix3(1.0f, 0.0f, 0.0f,
                              0.0f, cos(front_angle), sin(front_angle),
                              0.0f, -sin(front_angle), cos(front_angle)) *
                      Matrix3(rotate_sin, 0, rotate_cos,
                              0, 1, 0,
                              -rotate_cos, 0, rotate_sin);
        for (int i = 0; i < (int) indices.size(); i++) {
            indices[i] = std::make_pair((mat * particles[i].position).z, i);
        }
        std::sort(indices.begin(), indices.end());

        for (int i = 0; i < (int) indices.size(); i++) {
            const int index = indices[i].second;
            Vector3 tracker = particles[index].position;

            tracker = (tracker - center) * 0.7f;
            tracker = mat * tracker;
            tracker += center;

            Vector2 coord(tracker.x, tracker.y);
            Vector3 bright_color = particles[index].color;
            Vector3 color = lerp(exp(-occlusion[index]), bright_color * 0.5f, bright_color);
            float alpha = 0.1f;
            int x = (int) round(coord.x * buffer.get_width());
            int y = (int) round(coord.y * buffer.get_height());
            if (buffer.inside(x, y)) {
                buffer[x][y] = lerp(alpha, buffer[x][y], color);
            }
        }
    }

TC_NAMESPACE_END
