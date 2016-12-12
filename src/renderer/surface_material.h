#pragma once

#include "math/linalg.h"
#include <boost/property_tree/ptree_fwd.hpp>
#include "texture.h"
#include "common/meta.h"
#include "volume_material.h"

TC_NAMESPACE_BEGIN

enum class SurfaceScatteringFlags {
	delta = 1 << 0,
	non_delta = 1 << 1,
	emit = 1 << 2,
	index_matched = 1 << 3,
	entering = 1 << 4,
	leaving = 1 << 5,
};

typedef int SurfaceEvent;

class SurfaceEventClassifier {
public:
	static bool is_delta(const SurfaceEvent &event) {
		return (event & (int)SurfaceScatteringFlags::delta) != 0;
	}
	static bool is_entering(const SurfaceEvent &event) {
		return (event & (int)SurfaceScatteringFlags::entering) != 0;
	}
	static bool is_leaving(const SurfaceEvent &event) {
		return (event & (int)SurfaceScatteringFlags::leaving) != 0;
	}
	static bool is_emit(const SurfaceEvent &event) {
		return (event & (int)SurfaceScatteringFlags::emit) != 0;
	}
	static bool is_index_matched(const SurfaceEvent &event) {
		return (event & (int)SurfaceScatteringFlags::index_matched) != 0;
	}
};

class SurfaceMaterial {
protected:
	std::shared_ptr<Texture> color_sampler;
	std::shared_ptr<VolumeMaterial> internal_material = nullptr;
public:
	SurfaceMaterial() {
		internal_material = nullptr;
	}

	virtual void set_internal_material(const std::shared_ptr<VolumeMaterial> &vol) {
		this->internal_material = vol;
	}

	virtual VolumeMaterial const *get_internal_material() {
		return internal_material.get();
	}

	virtual bool is_index_matched() const {
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
		SurfaceEvent &event, const Vector2 &uv) const {
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

	static std::shared_ptr<Texture> get_color_sampler(const Config &config, const string &name) {
		if (config.has_key(name + "_map")) {
			return AssetManager::get_asset<Texture>(config.get_int(name + "_map"));
		}
		else if (config.has_key(name)) {
			Vector3 color = config.get_vec3(name);
			return create_initialized_instance<Texture>("const", Config().set("value", color));
		}
		else {
			return nullptr;
		}
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

	virtual real get_importance(const Vector2 &uv) const {
		return luminance(color_sampler->sample(uv));
	}


public:
	static Vector3 reflect(const Vector3 &in) { // Note: in and reflected are both from origin to outside
		return Vector3(-in.x, -in.y, in.z);
	}
};

TC_INTERFACE(SurfaceMaterial);

TC_NAMESPACE_END
