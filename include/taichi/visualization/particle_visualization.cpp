#include "particle_visualization.h"
#include <taichi/math/array_3d.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(ParticleRenderer, "particle_renderer");

class ParticleShadowMapRenderer : public ParticleRenderer {
private:
	Vector3 light_direction;
	real shadow_map_resolution;
	Matrix3 light_transform;
	real ambient_light;
	real shadowing;
	real alpha;
public:
	ParticleShadowMapRenderer() {}

	virtual void initialize(const Config &config) override {
		shadow_map_resolution = config.get_real("shadow_map_resolution");
		light_direction = config.get_vec3("light_direction");
		ambient_light = config.get("ambient_light", 0.0f);
		shadowing = config.get("shadowing", 1.0f);
		alpha = config.get("alpha", 1.0f);
		light_direction = normalized(light_direction);
		Vector3 u = abs(light_direction.y) > 0.99f ? Vector3(1, 0, 0) :
			normalized(glm::cross(light_direction, Vector3(0, 1, 0)));
		Vector3 v = normalized(glm::cross(u, light_direction));
		light_transform = glm::transpose(Matrix3(u, v, light_direction));
	}

	virtual void render(ImageBuffer<Vector3> &buffer, const std::vector<RenderParticle> &particles) const override {
		if (particles.empty()) {
			return;
		}
		Vector2 uv_lowerbound(2000 * shadow_map_resolution);
		Vector2 uv_upperbound(-2000 * shadow_map_resolution);

		std::vector <std::pair<real, int>> indices(particles.size());
		for (int i = 0; i < (int)indices.size(); i++) {
			indices[i] = std::make_pair(-glm::dot(light_direction, particles[i].position), i);
			Vector3 transformed_coord = light_transform * particles[i].position;
			Vector2 uv(transformed_coord.x, transformed_coord.y);
			uv_lowerbound.x = std::min(uv_lowerbound.x, uv.x);
			uv_lowerbound.y = std::min(uv_lowerbound.y, uv.y);
			uv_upperbound.x = std::max(uv_upperbound.x, uv.x);
			uv_upperbound.y = std::max(uv_upperbound.y, uv.y);
		}
		std::sort(indices.begin(), indices.end());
		Vector2 res = (uv_upperbound - uv_lowerbound) / shadow_map_resolution;
		Array2D<real> occlusion_buffer((int)std::ceil(res.x) + 1, (int)std::ceil(res.y) + 1, 1.0f);
		real shadow_map_scaling = 1.0f / shadow_map_resolution;
		std::vector<real> occlusion(particles.size());

		for (int i = 0; i < (int)indices.size(); i++) {
			const int index = indices[i].second;
			Vector3 transformed_coord = light_transform * particles[index].position;
			Vector2 uv(transformed_coord.x, transformed_coord.y);
			uv = shadow_map_scaling * (uv - uv_lowerbound);
			int int_x = (int)(uv.x);
			int int_y = (int)(uv.y);
			real occ = 0.0f;
			if (occlusion_buffer.inside(uv)) {
				occlusion_buffer[int_x][int_y] *= (1.0f - shadowing * particles[index].color.w);
				occ = occlusion_buffer.sample(uv);
			}
			occlusion[index] = std::max(ambient_light, occ);
		}

		for (int i = 0; i < (int)indices.size(); i++) {
			real dist = -glm::dot(camera->get_dir(), particles[i].position - camera->get_origin());
			indices[i] = std::make_pair(dist, i);
		}
		std::sort(indices.begin(), indices.end());
		for (int i = 0; i < (int)indices.size(); i++) {
			const int index = indices[i].second;
			auto &p = particles[index];
			real dist = -glm::dot(camera->get_dir(), particles[index].position - camera->get_origin());
			if (dist >= 0) {
				continue;
			}
			auto direction = normalized(p.position - camera->get_origin());
			real u, v;
			camera->get_pixel_coordinate(direction, u, v);
			int int_u = (int)(u * buffer.get_width());
			int int_v = (int)(v * buffer.get_height());
			if (buffer.inside(int_u, int_v)) {
				Vector3 color(p.color.x, p.color.y, p.color.z);
				real alpha = p.color.w * this->alpha;
				buffer[int_u][int_v] = lerp(alpha, buffer[int_u][int_v], color * occlusion[index]);
			}
		}
	}
};

TC_IMPLEMENTATION(ParticleRenderer, ParticleShadowMapRenderer, "shadow_map");

std::shared_ptr<Texture> rasterize_render_particles(const Config &config, const std::vector<RenderParticle> &particles) {
	Vector3i resolution = config.get_vec3i("resolution");
	Array3D<Vector3> array(resolution[0], resolution[1], resolution[2], Vector3(0));
	auto kernel = [](const Vector3 &d) {
		return std::abs(d.x) * std::abs(d.y) * std::abs(d.z);
	};
	for (auto const &p : particles) {
		for (auto &ind : array.get_rasterization_region(p.position, 1)) {
			Vector3 color(p.color.x, p.color.y, p.color.z);
			array[ind] += color * kernel(p.position - ind.get_pos());
		}
	}
	Config cfg;
	cfg.set("array_ptr", &array);
	cfg.print_all();
	auto tex = create_initialized_instance<Texture>("array3d", cfg);
	P(nullptr != tex.get());
	return tex;
}

TC_NAMESPACE_END
