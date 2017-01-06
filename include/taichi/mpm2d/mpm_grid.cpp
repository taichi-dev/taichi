#include "mpm_grid.h"
#include <stb_image.h>

TC_NAMESPACE_BEGIN

vec2 particle_offsets[]{ vec2(0.25f, 0.25f), vec2(0.75f, 0.25f), vec2(0.25f, 0.75f), vec2(0.75f, 0.75f) };
long long MPMParticle::instance_count = 0;

void Grid::apply_boundary_conditions(const LevelSet2D & levelset) {
    if (levelset.get_width() > 0) {
        for (auto &ind : boundary_normal.get_region()) {
            vec2 v = velocity[ind], n = levelset.get_normalized_gradient(Vector2(ind.i + 0.5f, ind.j + 0.5f));
            float phi = levelset[ind];
            if (phi > 1) continue;
            else if (phi > 0) { // 0~1
                float pressure = max(-glm::dot(v, n), 0.0f);
                float mu = levelset.friction;
                if (mu < 0) { // sticky
                    velocity[ind] = Vector2(0.0f);
                }
                else {
                    vec2 t = vec2(-n.y, n.x);
                    float friction = -clamp(glm::dot(t, v), -mu * pressure, mu * pressure);
                    velocity[ind] = v + n * pressure + t * friction;
                }
            }
            else if (phi <= 0) {
                velocity[ind] = Vector2(0.0f);
            }
        }
    }
}

TC_NAMESPACE_END
