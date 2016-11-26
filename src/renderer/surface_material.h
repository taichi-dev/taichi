#pragma once

#include "math/linalg.h"
#include <boost/property_tree/ptree_fwd.hpp>
#include "texture.h"
#include "common/meta.h"
#include "volume.h"

TC_NAMESPACE_BEGIN

    enum class SurfaceScatteringEvent {
        delta = 1 << 0,
        non_delta = 1 << 1,
        emit = 1 << 2
    };

    class SurfaceMaterial {
    protected:
        std::shared_ptr<Texture> color_sampler;
        std::shared_ptr<VolumeMaterial> internal_material = nullptr;
    public:
        using ScatteringEvent = SurfaceScatteringEvent;

        SurfaceMaterial() {
			internal_material = nullptr;
		}

		virtual void set_internal_material(const std::shared_ptr<VolumeMaterial> &vol) {
			this->internal_material = vol;
		}

		virtual VolumeMaterial const *get_internal_material() {
			return internal_material.get();
		}

		virtual bool is_index_matched() {
			return false;
		}

        virtual void initialize(const Config &config) {}

        virtual void initialize(boost::property_tree::ptree &pt) {
        }

        virtual Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const {
            assert_info(false, "Not implemented");
            return Vector3(0, 0, 0);
        }

        virtual void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir, Vector3 &f, real &pdf,
                            ScatteringEvent &event, const Vector2 &uv) const {
            assert_info(false, "Not implemented");
        }

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const {
            return 0.0f;
        };

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const {
            return Vector3(0.0f);
        }

        virtual void set_color_sampler(const std::shared_ptr<Texture> color_sampler) {
            this->color_sampler = color_sampler;
        }

        virtual void set_color(const Vector3 &color) {
            this->color_sampler = create_initialized_instance<Texture>("const", Config().set("value", color));
        }

        virtual real get_intensity(const Vector2 &uv) {
            return luminance(color_sampler->sample(uv));
        }

        virtual bool is_delta() const {
            return false;
        }

        virtual bool is_emissive() const {
            return false;
        }

        static bool is_delta(const ScatteringEvent &event) {
            return ((int) event & (int) ScatteringEvent::delta) != 0;
        }

    public:
        static Vector3 reflect(const Vector3 &in) { // Note: in and reflected are both from origin to outside
            return Vector3(-in.x, -in.y, in.z);
        }
    };

    TC_INTERFACE(SurfaceMaterial);

TC_NAMESPACE_END
