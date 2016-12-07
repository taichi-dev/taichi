#pragma once

#include "io/importer.h"
#include "geometry_primitives.h"
#include "common/config.h"
#include "common/meta.h"

TC_NAMESPACE_BEGIN

    class Camera {
    public:
        virtual void initialize(const Config &config) {};

        virtual Ray sample(Vector2 offset, Vector2 size, real u, real v) {
            assert_info(false, "no impl");
            return Ray(Vector3(0), Vector3(0));
        };

        virtual Ray sample(Vector2 offset, Vector2 size) {
            return sample(offset, size, rand(), rand());
        }

        Vector3 get_origin() {
            return multiply_matrix4(transform, origin, 1);
        }

        Vector3 get_dir() {
            return multiply_matrix4(transform, dir, 0);
        }

        virtual void get_pixel_coordinate(Vector3 dir, real &u, real &v) {
        }

        virtual real get_pixel_scaling() {
            return 1.0f;
        }

		int get_width() const {
			return width;
		}

		int get_height() const {
			return height;
		}

    protected:
        Vector3 origin, look_at, up, right, dir;
        Matrix4 transform;
		int width, height;

        void set_dir_and_right() {
            this->dir = normalize(look_at - origin);
            this->up = normalize(up - dir * dot(dir, up));
            this->right = cross(dir, up);
        }

        Vector2 random_offset(Vector2 offset, Vector2 size, real u, real v) {
            return Vector2(offset.x - 0.5f + u * size.x, offset.y - 0.5f + v * size.y);
        }
    };

    TC_INTERFACE(Camera);

TC_NAMESPACE_END

