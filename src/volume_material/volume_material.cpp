#include <taichi/visual/volume_material.h>
#include <taichi/visual/texture.h>
#include <taichi/math/array_3d.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(VolumeMaterial, "volume_material")
TC_IMPLEMENTATION(VolumeMaterial, VolumeMaterial, "homogeneous");

class VacuumVolumeMaterial : public VolumeMaterial {
public:
	virtual void initialize(const Config &config) override {
		VolumeMaterial::initialize(config);
		this->volumetric_scattering = 0.0f;
		this->volumetric_absorption = 0.0f;
	}
	virtual real get_attenuation(real dist) const override {
		return 1.0f;
	}

	virtual real unbiased_sample_attenuation(const Vector3 &start, const Vector3 &end, StateSequence &rand) const override {
		return 1.0f;
	}

	virtual VolumeEvent sample_event(StateSequence &rand, const Ray &ray) const override {
		error("invalid");
		return VolumeEvent::absorption;
	}
	virtual bool is_vacuum() const override {
		return true;
	}
};

TC_IMPLEMENTATION(VolumeMaterial, VacuumVolumeMaterial, "vacuum");

class VoxelVolumeMaterial : public VolumeMaterial {
protected:
	Array3D<real> voxels;
	std::shared_ptr<Texture> tex;
	Vector3i resolution;
	real maximum;

public:
	virtual void initialize(const Config &config) override {
		VolumeMaterial::initialize(config);
		this->volumetric_scattering = config.get_real("scattering");
		this->volumetric_absorption = config.get_real("absorption");
		this->resolution = config.get_vec3i("resolution");
		this->tex = AssetManager::get_asset<Texture>(config.get_int("tex"));
		voxels.initialize(resolution.x, resolution.y, resolution.z, 1.0f);
		maximum = 0.0f;
		Vector3 inv = Vector3(1.0f) / Vector3(resolution);
		for (auto &ind : voxels.get_region()) {
			voxels[ind] = tex->sample(ind.get_pos() * inv).x;
			maximum = std::max(maximum, voxels[ind]);
		}
	}

	virtual real sample_free_distance(StateSequence &rand, const Ray &ray) const override {
		int counter = 0;
		real kill;
		real dist = 0.0f;
		real tot = volumetric_scattering + volumetric_absorption;
		do {
			counter += 1;
			if (counter > 10000) {
				printf("Warning: path too long\n");
				break;
			}
			dist += -log(1 - rand()) / (maximum * tot);
			Vector3 pos = ray.orig + ray.dir * dist;
			pos = multiply_matrix4(world2local, pos, 1.0f);
			if (pos.x < 0 || pos.x >= 1 || pos.y < 0 || pos.y >= 1 || pos.z < 0 || pos.z >= 1) {
				// Outside the texture
				dist = std::numeric_limits<real>::infinity();
				break;
			}
			kill = voxels.sample_relative_coord(pos);
		} while (maximum * rand() > kill && dist < ray.dist);
		return dist;
	}

	virtual real unbiased_sample_attenuation(const Vector3 &start, const Vector3 &end, StateSequence &rand) const override {
		auto dir = normalized(end - start);
		return sample_free_distance(rand, Ray(start, dir)) >= glm::length(end - start);
	}
};

VolumeStack::VolumeStack() {
	static std::shared_ptr<VolumeMaterial> vacuum = create_instance<VolumeMaterial>("vacuum");
	stack.push_back(vacuum.get());
}

TC_IMPLEMENTATION(VolumeMaterial, VoxelVolumeMaterial, "voxel");

TC_NAMESPACE_END
