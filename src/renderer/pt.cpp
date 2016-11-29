#include "renderer.h"
#include "sampler.h"
#include "bsdf.h"
#include "markov_chain.h"
#include "volume.h"
#include "system/threading.h"

TC_NAMESPACE_BEGIN
struct PathContribution {
	float x, y;
	Vector3 c;

	PathContribution() {};

	PathContribution(float x, float y, Vector3 c) :
		x(x), y(y), c(c) {}
};

// TODO: we do need a light source class to unify envmap and mesh light...

class PathTracingRenderer : public Renderer {
public:
	virtual void initialize(const Config &config) override;

	void render_stage() override;

	virtual ImageBuffer<Vector3> get_output() override;

protected:

	VolumeMaterial volume;

	bool russian_roulette;

	Vector3 calculate_direct_lighting(const Vector3 &in_dir, const IntersectionInfo &info, const BSDF &bsdf,
		StateSequence &rand, VolumeStack &stack) {
		Vector3 acc(0);
		if (scene->envmap) {
			acc += calculate_direct_lighting_env(in_dir, info, bsdf, rand, stack);
		}
		if (!full_direct_lighting) {
			real triangle_pdf;
			Triangle &tri = scene->sample_triangle_light_emission(rand(), triangle_pdf);
			if (tri.get_relative_location_to_plane(info.pos) > 0) {
				acc += calculate_direct_lighting(in_dir, info, bsdf, rand, tri, stack) / triangle_pdf;
			}
		}
		else {
			for (auto &tri : scene->emissive_triangles) {
				if (tri.get_relative_location_to_plane(info.pos) > 0) {
					acc += calculate_direct_lighting(in_dir, info, bsdf, rand, tri, stack);
				}
			}
		}
		return acc;
	}

	Vector3 calculate_direct_lighting_env(const Vector3 &in_dir, const IntersectionInfo &info, const BSDF &bsdf,
		StateSequence &rand, VolumeStack &stack) {
		Vector3 acc(0);
		{
			int samples = direct_lighting_bsdf + direct_lighting_light;
			assert_info(samples != 0, "Sum of direct_lighting_bsdf and direct_lighting_light should not be 0.");
			for (int i = 0; i < samples; i++) {
				bool sample_bsdf = i < direct_lighting_bsdf;
				if (!sample_bsdf && bsdf.is_delta()) {
					// Light sampling doesn't work in case of delta BSDF.
					continue;
				}
				Vector3 out_dir;
				Vector3 f;
				real bsdf_p;
				SurfaceEvent event;
				real light_p = 1.0f / 4 / pi;
				Vector3 emission;
				if (sample_bsdf) { // Sample BSDF
					bsdf.sample(in_dir, rand(), rand(), out_dir, f, bsdf_p, event);
					emission = scene->envmap->sample_illum(out_dir);
					if (SurfaceEventClassifier::is_index_matched(event)) {
						continue;
					}
				}
				else { // Sample light source
					if (envmap_is) {
						real pdf;
						out_dir = scene->envmap->sample_direction(rand, pdf, emission);
						light_p *= pdf;
					}
					else {
						out_dir = sample_sphere(rand(), rand());
						emission = scene->envmap->sample_illum(out_dir);
					}
					if (max_component(bsdf.evaluate(in_dir, out_dir)) == 0) {
						continue;
					}
				}
				Ray ray(info.pos + out_dir * 1e-3f, out_dir, 0);
				IntersectionInfo test_info;
				Vector3 att = get_attenuation(stack, ray, rand, test_info);
				if (test_info.intersected) {
					continue;
				}
				if (max_component(att) == 0.0f) {
					continue;
				}

				if (!sample_bsdf) { // Sample light source
					f = bsdf.evaluate(in_dir, out_dir);
					bsdf_p = bsdf.probability_density(in_dir, out_dir);
				}
				real co = abs(dot(ray.dir, info.normal));
				Vector3 dist = test_info.pos - info.pos;
				const Vector3 throughput = emission * co * f * att;
				if (sample_bsdf) {
					if (SurfaceEventClassifier::is_delta(event)) {
						acc += 1 / (direct_lighting_bsdf * bsdf_p) * throughput;
					}
					else {
						acc += 1 / (direct_lighting_bsdf * bsdf_p +
							direct_lighting_light * light_p) * throughput;
					}
				}
				else {
					acc += 1 / (direct_lighting_bsdf * bsdf_p +
						direct_lighting_light * light_p) * throughput;
				}
			}
			return acc;
		}
	}

	Vector3 calculate_direct_lighting(const Vector3 &in_dir, const IntersectionInfo &info, const BSDF &bsdf,
		StateSequence &rand, const Triangle &tri, VolumeStack &stack);

	Vector3 calculate_volumetric_direct_lighting(const Vector3 &in_dir, const Vector3 &orig,
		StateSequence &rand, VolumeStack &stack);

	PathContribution get_path_contribution(StateSequence &rand) {
		Vector2 offset(rand(), rand());
		Vector2 size(1.0f / width, 1.0f / height);
		Ray ray = camera->sample(offset, size);
		Vector3 color = trace(ray, rand);
		if (luminance_clamping > 0 && luminance(color) > luminance_clamping) {
			color = luminance_clamping / luminance(color) * color;
		}
		return PathContribution(offset.x, offset.y, color);
	}

	Vector3 trace(Ray ray, StateSequence &rand);

	virtual void write_path_contribution(const PathContribution &cont, real scale = 1.0f) {
		auto x = clamp(cont.x, 0.0f, 1.0f - 1e-7f);
		auto y = clamp(cont.y, 0.0f, 1.0f - 1e-7f);
		if (!is_normal(cont.c)) {
			P(cont.c);
			return;
		}
		accumulator.accumulate(int(x * width), int(y * height), cont.c * scale);
	}

	virtual Vector3 get_attenuation(VolumeStack stack, Ray ray, StateSequence &rand, IntersectionInfo &last_intersection) {
		Vector3 att(1.0f);

		while (true) {
			IntersectionInfo info = sg->query(ray);
			if (!info.intersected) {
				// no intersection (usually bsdf sampling)
				last_intersection = info;
				break;
			}
			BSDF bsdf(scene, info);
			if (bsdf.is_emissive()) {
				// light source...
				last_intersection = info;
				break;
			}
			const Vector3 in_dir = -ray.dir;
			if (stack.size() == 0) {
				// TODO: this is a bug...
				return Vector3(0.0f);
			}
			const VolumeMaterial &vol = *stack.top();
			if (bsdf.is_index_matched()) {
				att *= bsdf.evaluate(in_dir, -in_dir) * bsdf.cos_theta(-in_dir);
				att *= vol.unbiased_sample_attenuation(ray.orig, info.pos, rand);
				//P(vol.unbiased_sample_attenuation(ray.orig, info.pos, rand));
				//P(att);
				ray = Ray(info.pos + ray.dir * 1e-3f, ray.dir);
				if (bsdf.is_entering(in_dir)) {
					assert(bsdf.get_internal_material() != nullptr);
					stack.push(bsdf.get_internal_material());
				}
				else {
					assert(stack.top() == bsdf.get_internal_material());
					stack.pop();
				}
			}
			else {
				// We do not handle non-index matched case here.
				return Vector3(0.0f);
			}
			//P(stack.size());
		}
		return att;
	}

	bool direct_lighting;
	int direct_lighting_bsdf;
	int direct_lighting_light;
	ImageAccumulator<Vector3> accumulator;
	std::shared_ptr<Sampler> sampler;
	long long index;
	real luminance_clamping;
	bool full_direct_lighting;
	bool envmap_is;
};

void PathTracingRenderer::initialize(const Config &config) {
	Renderer::initialize(config);
	this->direct_lighting = config.get("direct_lighting", true);
	this->direct_lighting_light = config.get("direct_lighting_light", 1);
	this->direct_lighting_bsdf = config.get("direct_lighting_bsdf", 1);
	this->sampler = create_instance<Sampler>(config.get("sampler", "prand"));
	this->luminance_clamping = config.get("luminance_clamping", 0.0f);
	this->full_direct_lighting = config.get("full_direct_lighting", false);
	this->accumulator = ImageAccumulator<Vector3>(width, height);
	this->russian_roulette = config.get("russian_roulette", true);
	this->envmap_is = config.get("envmap_is", true);
	index = 0;
}

void PathTracingRenderer::render_stage() {
	int samples = width * height;
	auto task = [&](int i) {
		index = index + i;
		RandomStateSequence rand(sampler, index);
		auto cont = get_path_contribution(rand);
		write_path_contribution(cont);
	};
	ThreadedTaskManager::run(task, 0, samples, num_threads);
	index += samples;
}

ImageBuffer<Vector3> PathTracingRenderer::get_output() {
	return accumulator.get_averaged();
}

Vector3 PathTracingRenderer::calculate_direct_lighting(const Vector3 &in_dir, const IntersectionInfo &info,
	const BSDF &bsdf,
	StateSequence &rand, const Triangle &tri, VolumeStack &stack) {
	// MIS between bsdf and light sampling.
	Vector3 acc(0);
	{
		int samples = direct_lighting_bsdf + direct_lighting_light;
		assert_info(samples != 0, "Sum of direct_lighting_bsdf and direct_lighting_light should not be 0.");
		for (int i = 0; i < samples; i++) {
			bool sample_bsdf = i < direct_lighting_bsdf;
			if (!sample_bsdf && bsdf.is_delta()) {
				// Light sampling doesn't work in case of delta BSDF.
				continue;
			}
			Vector3 out_dir;
			Vector3 f;
			real bsdf_p;
			SurfaceEvent event;
			if (sample_bsdf) { // Sample BSDF
				bsdf.sample(in_dir, rand(), rand(), out_dir, f, bsdf_p, event);
				if (SurfaceEventClassifier::is_index_matched(event)) {
					continue;
				}
			}
			else { // Sample light source
				Vector3 pos = tri.sample_point(rand(), rand());
				Vector3 dist = pos - info.pos;
				out_dir = normalize(dist);
			}
			Ray ray(info.pos + out_dir * 1e-3f, out_dir, 0);
			IntersectionInfo test_info;
			Vector3 att = get_attenuation(stack, ray, rand, test_info);
			if (!test_info.intersected || test_info.triangle_id != tri.id) {
				continue;
			}
			if (max_component(att) == 0.0f) {
				continue;
			}

			if (!sample_bsdf) { // Sample light source
				f = bsdf.evaluate(in_dir, out_dir);
				bsdf_p = bsdf.probability_density(in_dir, out_dir);
			}
			real co = abs(dot(ray.dir, info.normal));
			real c = abs(dot(ray.dir, tri.normal));
			Vector3 dist = test_info.pos - info.pos;
			real light_p = dot(dist, dist) / (tri.area * c);
			BSDF light_bsdf(scene, test_info);
			const Vector3 emission = light_bsdf.evaluate(test_info.normal, -out_dir);
			const Vector3 throughput = emission * co * f * att;
			if (sample_bsdf) {
				if (sample_bsdf && SurfaceEventClassifier::is_delta(event)) {
					acc += 1 / (direct_lighting_bsdf * bsdf_p) * throughput;
				}
				else {
					acc += 1 / (direct_lighting_bsdf * bsdf_p +
						direct_lighting_light * light_p) * throughput;
				}
			}
			else {
				acc += 1 / (direct_lighting_bsdf * bsdf_p +
					direct_lighting_light * light_p) * throughput;
			}
		}
		return acc;
	}
}

Vector3 PathTracingRenderer::calculate_volumetric_direct_lighting(const Vector3 &in_dir, const Vector3 &orig,
	StateSequence &rand, VolumeStack &stack) {
	Vector3 lighting(0);
	real triangle_pdf;
	Triangle &tri = scene->sample_triangle_light_emission(rand(), triangle_pdf);
	// MIS
	for (int i = 0; i < 2; i++) {
		bool sample_phase = i == 0;
		Vector3 out_dir;
		if (sample_phase) {
			// Sample phase function
			out_dir = stack.top()->sample_phase(rand);
		}
		else {
			// Sample light source
			Vector3 pos = tri.sample_point(rand(), rand());
			Vector3 dist = pos - orig;
			out_dir = normalize(dist);
		}
		Ray out_ray(orig, out_dir);

		real phase_pdf = 1 / (4 * pi);
	
		real c = abs(dot(out_ray.dir, tri.normal));
		IntersectionInfo info;
		Vector3 att = get_attenuation(stack, out_ray, rand, info);
		//P(att);
		Vector3 dist = info.pos - orig;
		real light_pdf = dot(dist, dist) / (tri.area * c);

		real mis_weight = 1.0f / (phase_pdf + light_pdf);
		Vector3 f(1.0f / (4.0f * pi));
		if (info.intersected && info.front && info.triangle_id == tri.id) {
			const BSDF light_bsdf(scene, info);
			const Vector3 emission = light_bsdf.evaluate(info.normal, -out_dir);
			const Vector3 throughput = f * emission * att * mis_weight;
			lighting += throughput;
		}
	}
	return lighting / triangle_pdf;
}

Vector3 PathTracingRenderer::trace(Ray ray, StateSequence &rand) {
	Vector3 ret(0);
	Vector3 importance(1);
	VolumeStack stack;
	bool surface_dl_counted = false;
	if (scene->get_atmosphere_material()) {
		stack.push(scene->get_atmosphere_material().get());
	}
	for (int depth = 1; depth <= max_path_length; depth++) {
		if (stack.size() == 0) {
			// What's going on here...
			P(stack.size());
			break;
		}
		const VolumeMaterial &volume = *stack.top();
		IntersectionInfo info = sg->query(ray);
		real safe_distance = volume.sample_free_distance(rand);
		Vector3 f(1.0f);
		Ray out_ray;
		if (!info.intersected) {
			if (scene->envmap && (depth == 1 || !direct_lighting)) {
				ret += importance * scene->envmap->sample_illum(ray.dir);
			}
			break;
		}
		if (direct_lighting) {
			// TOOD: add solid angle IS?
		}
		if (info.dist < safe_distance) {
			// Safely travels to the next surface...

			// Attenuation
			Vector3 att(volume.unbiased_sample_attenuation(ray.orig, info.pos, rand));
			importance *= att;

			BSDF bsdf(scene, info);
			const Vector3 in_dir = -ray.dir;
			if (bsdf.is_emissive()) {
				//assert(stack.size() == 2);
				bool count = !surface_dl_counted && info.front && (depth == 1 || !direct_lighting);
				if (count && path_length_in_range(depth)) {
					ret += importance * bsdf.evaluate(info.normal, in_dir);
				}
				break;
			}
			if (!surface_dl_counted && direct_lighting && path_length_in_range(depth + 1)) {
				ret += importance * calculate_direct_lighting(in_dir, info, bsdf, rand, stack);
			}
			surface_dl_counted = true;
			real pdf;
			SurfaceEvent event;
			Vector3 out_dir;
			bsdf.sample(in_dir, rand(), rand(), out_dir, f, pdf, event);
			surface_dl_counted = false;
			if (bsdf.is_entering(in_dir) && !bsdf.is_entering(out_dir)) {
				if (bsdf.get_internal_material() != nullptr)
					stack.push(bsdf.get_internal_material());
			}
			if (bsdf.is_entering(out_dir) && !bsdf.is_entering(in_dir)) {
				if (bsdf.get_internal_material() != nullptr) {
					stack.pop();
				}
			}
			out_ray = Ray(info.pos, out_dir, 1e-5f);
			real c = abs(glm::dot(out_dir, info.normal));
			if (pdf < 1e-10f) {
				break;
			}
			f *= c / pdf;
		}
		else if (volume.sample_event(rand) == VolumeEvent::scattering) {
			// Volumetric scattering
			const Vector3 orig = ray.orig + ray.dir * safe_distance;
			const Vector3 in_dir = -ray.dir;
			if (direct_lighting && path_length_in_range(depth + 1)) {
				//P(stack.size());
				ret += importance * calculate_volumetric_direct_lighting(in_dir, orig, rand, stack);
			}
			Vector3 out_dir = volume.sample_phase(rand);
			out_ray = Ray(orig, out_dir, 1e-5f);
			f = Vector3(1.0f);
		}
		else {
			// Volumetric absorption
			break;
		}
		ray = out_ray;
		importance *= f;
		if (russian_roulette) {
			real p = luminance(importance);
			if (p <= 1) {
				if (rand() < p) {
					importance *= 1.0f / p;
				}
				else {
					break;
				}
			}
		}
	}
	return ret;
}

class PSSMLTMarkovChain : public MarkovChain {
public:
	real resolution_x, resolution_y;

	PSSMLTMarkovChain() : PSSMLTMarkovChain(0, 0) {}

	PSSMLTMarkovChain(real resolution_x, real resolution_y) : resolution_x(resolution_x), resolution_y(resolution_y) {
	}

	PSSMLTMarkovChain large_step() const {
		return PSSMLTMarkovChain(resolution_x, resolution_y);
	}

	PSSMLTMarkovChain mutate(real strength = 1.0f) const {
		PSSMLTMarkovChain result(*this);
		// Pixel location
		real delta_pixel = 2.0f / (resolution_x + resolution_y);
		result.get_state(2);
		result.states[0] = perturb(result.states[0], delta_pixel * strength, 0.1f * strength);
		result.states[1] = perturb(result.states[1], delta_pixel * strength, 0.1f * strength);
		// Events
		for (int i = 2; i < (int)result.states.size(); i++)
			result.states[i] = perturb(result.states[i], 1.0f / 1024.0f * strength, 1.0f / 64.0f * strength);
		return result;
	}

protected:
	inline static real perturb(const real value, const real s1, const real s2) {
		real result;
		real r = rand();
		if (r < 0.5f) {
			r = r * 2.0f;
			result = value + s2 * exp(-log(s2 / s1) * r);
		}
		else {
			r = (r - 0.5f) * 2.0f;
			result = value - s2 * exp(-log(s2 / s1) * r);
		}
		result -= floor(result);
		return result;
	}
};


class MCMCPTRenderer : public PathTracingRenderer {
protected:
	struct MCMCState {
		PSSMLTMarkovChain chain;
		PathContribution pc;
		real sc;

		MCMCState() {}

		MCMCState(const PSSMLTMarkovChain &chain, const PathContribution &pc, real sc) :
			chain(chain), pc(pc), sc(sc) {
		}
	};

	int estimation_rounds;
	MCMCState current_state;
	bool first_stage_done = false;
	real b;
	real large_step_prob;
	real mutation_strength;
	long long sample_count;
	ImageBuffer<Vector3> buffer;
public:
	ImageBuffer<Vector3> get_output() override {
		ImageBuffer<Vector3> output(width, height);
		float r = 1.0f / sample_count;
		for (auto &ind : output.get_region()) {
			output[ind] = buffer[ind] * r;
		};
		return output;
	}

	void initialize(const Config &config) override {
		PathTracingRenderer::initialize(config);
		large_step_prob = config.get("large_step_prob", 0.3f);
		estimation_rounds = config.get("estimation_rounds", 1);
		mutation_strength = config.get_real("mutation_strength");
		buffer.initialize(width, height, Vector3(0.0f));
		sample_count = 0;
	}

	real scalar_contribution_function(const PathContribution &pc) {
		return luminance(pc.c);
	}

	void write_path_contribution(const PathContribution &cont, real scale = 1.0f) override {
		if (!is_normal(cont.c)) {
			P(cont.c);
			return;
		}
		if (0 <= cont.x && cont.x <= 1 - eps && 0 <= cont.y && cont.y <= 1 - eps) {
			int ix = (int)floor(cont.x * width), iy = (int)floor(cont.y * height);
			this->buffer[ix][iy] += width * height * scale * cont.c;
		}
	}

	virtual void render_stage() override {
		if (!first_stage_done) {
			real total_sc = 0.0f;
			int num_samples = width * height * estimation_rounds;
			auto sampler = create_instance<Sampler>("prand");
			for (int i = 0; i < num_samples; i++) {
				auto rand = RandomStateSequence(sampler, i);
				total_sc += scalar_contribution_function(get_path_contribution(rand));
			}
			b = total_sc / num_samples;
			P(b);
			current_state.chain = PSSMLTMarkovChain((real)width, (real)height);
			auto rand = MCStateSequence(current_state.chain);
			current_state.pc = get_path_contribution(rand);
			current_state.sc = scalar_contribution_function(current_state.pc);
			first_stage_done = true;
		}
		MCMCState new_state;
		for (int k = 0; k < width * height; k++) {
			real is_large_step;
			if (rand() <= large_step_prob) {
				new_state.chain = current_state.chain.large_step();
				is_large_step = 1.0;
			}
			else {
				new_state.chain = current_state.chain.mutate(mutation_strength);
				is_large_step = 0.0;
			}
			auto rand = MCStateSequence(new_state.chain);
			new_state.pc = get_path_contribution(rand);
			new_state.sc = scalar_contribution_function(new_state.pc);
			double a = 1.0;
			if (current_state.sc > 0.0) {
				a = clamp(new_state.sc / current_state.sc, 0.0f, 1.0f);
			}
			// accumulate samples with mean value substitution and MIS
			if (new_state.sc > 0.0) {
				write_path_contribution(new_state.pc,
					real((a + is_large_step) / (new_state.sc / b + large_step_prob)));
			}
			if (current_state.sc > 0.0) {
				write_path_contribution(current_state.pc,
					real((1.0 - a) / (current_state.sc / b + large_step_prob)));
			}
			// conditionally accept the chain
			if (rand() <= a) {
				current_state = new_state;
			}
			sample_count += 1;
		}

	}
};

TC_IMPLEMENTATION(Renderer, PathTracingRenderer, "pt");
TC_IMPLEMENTATION(Renderer, MCMCPTRenderer, "mcmcpt");

TC_NAMESPACE_END

