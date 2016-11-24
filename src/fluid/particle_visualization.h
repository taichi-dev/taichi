#pragma once

#include "common/utils.h"
#include "visualization/image_buffer.h"
#include "common/config.h"

TC_NAMESPACE_BEGIN

    class ParticleShadowMapRenderer {
    private:
        Vector3 light_direction;
        real rotate_z;
        Vector3 center;
        int shadow_map_resolution;
        real shadowing;
    public:
        struct Particle {
            Vector3 position;
            Vector3 color;
            Particle() {}
            Particle(const Vector3 &position, const Vector3 &color) : position(position), color(color) {}
        };

        ParticleShadowMapRenderer() {
        }

        void set_shadowing(real shadowing) {
            this->shadowing = shadowing;
        }

        void set_center(const Vector3 &center) {
            this->center = center;
        }

        void set_rotate_z(real rotate_z) {
            this->rotate_z = rotate_z;
        }
        void set_shadow_map_resolution(int shadow_map_resolution) {
            this->shadow_map_resolution = shadow_map_resolution;
        }

        void set_light_direction(const Vector3 &light_direction) {
            this->light_direction = light_direction;
        }

        void render(ImageBuffer<Vector3> &buffer, const std::vector<Particle> particles);
    };

TC_NAMESPACE_END

