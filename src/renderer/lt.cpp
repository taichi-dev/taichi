#include <taichi/visual/renderer.h>
#include <taichi/visual/sampler.h>
#include <taichi/visual/bsdf.h>

TC_NAMESPACE_BEGIN

struct PathContribution {
	float x, y;
	Vector3 c;

	PathContribution() {};

	PathContribution(float x, float y, Vector3 c) :
		x(x), y(y), c(c) {}
};

class LTRenderer : public Renderer {
protected:
	std::shared_ptr<Sampler> sampler;
	ImageBuffer<Vector3> buffer;
	long long photon_counter;
	bool volumetric;

public:
	virtual void initialize(const Config &config) {
		Renderer::initialize(config);
		this->sampler = create_instance<Sampler>(config.get("sampler", "prand"));
		this->volumetric = config.get("volumetric", true);
		this->buffer.initialize(width, height, Vector3(0.0f));
		this->photon_counter = 0;
	}

	virtual void render_stage() {
		int num_photons_per_stage = width * height;
		for (int i = 0; i < num_photons_per_stage; i++) {
			auto state_sequence = RandomStateSequence(sampler, photon_counter);
			trace_photon(state_sequence);
			photon_counter += 1;
		}
	}

	ImageBuffer<Vector3> get_output() {
		ImageBuffer<Vector3> output(width, height);
		float r = 1.0f / photon_counter;
		for (auto &ind : output.get_region()) {
			output[ind] = buffer[ind] * r;
		}
		return output;
	}

	virtual void write_path_contribution(const PathContribution &cont, real scale = 1.0f) {
		if (0 <= cont.x && cont.x <= 1 - eps && 0 <= cont.y && cont.y <= 1 - eps) {
			int ix = (int)floor(cont.x * width), iy = (int)floor(cont.y * height);
			this->buffer[ix][iy] += width * height * scale * cont.c;
		}
	}

	void connect_to_camera(const Vector3 &pos, const Vector3 &normal, const Vector3 &flux,
		const BSDF &bsdf, const Vector3 in_dir) {
		real px, py;
		camera->get_pixel_coordinate(normalized(pos - camera->get_origin()), px, py);
		if (!(px < 0 || px > 1 || py < 0 || py > 1)) {
			Vector3 out_dir = normalized(camera->get_origin() - pos);
			auto test_ray = Ray(pos, out_dir);
			sg->query(test_ray);
			Vector3d d0 = pos - camera->get_origin();
			const double dist2 = dot(d0, d0);
			if (test_ray.dist > sqrt(dist2) - 1e-4f) {
				d0 = normalized(d0);
				const double c = dot(d0, camera->get_dir());
				real scale = real(abs(dot(d0, normal) / dist2 / (c * c * c)) / camera->get_pixel_scaling());
				Vector3 co = bsdf.evaluate(in_dir, out_dir);
				write_path_contribution(PathContribution(px, py, flux * co), scale);
			}
		}
	}

	bool trace_photon(StateSequence &rand) { // returns visibility
		bool visible = false;
		real pdf;
		const Triangle &tri = scene->sample_triangle_light_emission(rand(), pdf);
		auto light_bsdf = BSDF(scene, tri.id);
		Vector3 pos = tri.sample_point(rand(), rand()),
			dir = light_bsdf.sample_direction(light_bsdf.get_geometry_normal(), rand(), rand());
		Vector3 flux = Vector3(1.0f / pdf) * tri.area;
		if (min_path_length <= 1) {
			connect_to_camera(pos, tri.normal, flux, light_bsdf, tri.normal);
		}
		flux = flux * light_bsdf.evaluate(tri.normal, dir) * pi;
		Ray ray(pos + dir * 1e-4f, dir, 0); // TODO: ... 1e-4f
		for (int depth = 1; depth + 1 <= max_path_length; depth++) {
			IntersectionInfo info = sg->query(ray);
			if (!info.intersected)
				break;
			Triangle &tri = scene->triangles[info.triangle_id];
			BSDF bsdf(scene, info);
			Vector3 in_dir = -ray.dir;
			Vector3 out_dir;
			Vector3 f;
			SurfaceEvent event;
			real pdf;
			if (bsdf.is_emissive()) {
				break;
			}
			if (!SurfaceEventClassifier::is_delta(event)) {
				// Connect to camera
				connect_to_camera(info.pos, info.normal, flux, bsdf, in_dir);
			}
			bsdf.sample(in_dir, rand(), rand(), out_dir, f, pdf, event);
			Vector3 color = f * bsdf.cos_theta(out_dir) / pdf;
			real p = max_component(color);
			if (p < 1 && rand() < p) {
				flux = (1.0f / p) * flux;
			}
			else {
				break;
			}
			ray = Ray(info.pos, out_dir);
			flux *= color;
		}
		return visible;
	}
};

TC_IMPLEMENTATION(Renderer, LTRenderer, "lt");

TC_NAMESPACE_END
