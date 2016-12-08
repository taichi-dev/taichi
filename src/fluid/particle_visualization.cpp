#include "particle_visualization.h"

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
		Vector2 uv_lowerbound(1e30f);
		Vector2 uv_upperbound(-1e30f);

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
		Array2D<real> occlusion_buffer(std::ceil(res.x) + 1, std::ceil(res.y) + 1, 1.0f);
		real shadow_map_scaling = 1.0f / shadow_map_resolution;
		std::vector<real> occlusion(particles.size());

		for (int i = 0; i < (int)indices.size(); i++) {
			const int index = indices[i].second;
			Vector3 transformed_coord = light_transform * particles[index].position;
			Vector2 uv(transformed_coord.x, transformed_coord.y);
			uv = shadow_map_scaling * (uv - uv_lowerbound);
			int int_x = (int)(uv.x);
			int int_y = (int)(uv.y);
			occlusion[index] = std::max(ambient_light, occlusion_buffer.sample(uv));
			occlusion_buffer[int_x][int_y] *= (1.0f - shadowing * particles[index].color.w);
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
			auto direction = normalized(p.position - camera->get_origin());
			real u, v;
			camera->get_pixel_coordinate(direction, u, v);
			int int_u = clamp((int)(u * buffer.get_width()), 0, buffer.get_width() - 1);
			int int_v = clamp((int)(v * buffer.get_height()), 0, buffer.get_height() - 1);
			Vector3 color(p.color.x, p.color.y, p.color.z);
			real alpha = p.color.w * this->alpha;
			if (buffer.inside(int_u, int_v))
				buffer[int_u][int_v] = lerp(alpha, buffer[int_u][int_v], color * occlusion[index]);
		}
	}
};

TC_IMPLEMENTATION(ParticleRenderer, ParticleShadowMapRenderer, "shadow_map");

TC_NAMESPACE_END
