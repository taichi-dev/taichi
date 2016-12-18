#include "bidirectional_renderer.h"
#include <taichi/visual/surface_material.h>

TC_NAMESPACE_BEGIN

// Check here for a more understandable version: http://www.ci.i.u-tokyo.ac.jp/~hachisuka/smallpssmlt.cpp
// TODO: long path sometimes explodes...

void BidirectionalRenderer::initialize(const Config &config) {
	Renderer::initialize(config);
	this->sampler = create_instance<Sampler>(config.get("sampler", "sobol"));
	this->luminance_clamping = config.get("luminance_clamping", 0.0f);
	this->buffer = ImageBuffer<Vector3>(width, height);
	this->max_eye_events = config.get("max_eye_events", 5);
	this->max_light_events = config.get("max_light_events", 5);
	this->max_eye_events = std::min(this->max_eye_events, this->max_path_length + 1);
	this->max_light_events = std::min(this->max_light_events, this->max_path_length + 1);
	this->stage_frequency = config.get("stage_frequency", 1);
	this->print_path_policy = config.get("print_path_policy", "bright");
	if (luminance_clamping == 0.0f) {
		this->print_path_policy = "none";
	}
}

void BidirectionalRenderer::trace(Path &path, Ray r, int depth, int max_depth, StateSequence &rand) {
	while (depth < max_depth) {
		IntersectionInfo info = sg->query(r);
		if (!info.intersected) {
			break;
		}
		BSDF bsdf(scene, info);
		Vertex v(info, bsdf);
		depth += 1;
		if (bsdf.is_emissive()) {
			// hit light source...
			v.event = (int)SurfaceScatteringFlags::emit;
			path.push_back(v);
			break;
		}
		else {
			v.in_dir = -r.dir;
			bsdf.sample(v.in_dir, rand(), rand(), v.out_dir, v.f, v.pdf, v.event);
			r = Ray(info.pos + v.out_dir * eps, v.out_dir); // TODO: fix here...
			path.push_back(v);
		}
	}
	while (depth < max_depth) {
		rand();
		rand();
		depth += 1;
	}
}

Path BidirectionalRenderer::trace_eye_path(StateSequence &rand) {
	Path result;
	result.reserve(max_eye_events);
	if (max_eye_events == 0) {
		return result;
	}
	Ray r = camera->sample(Vector2(0, 0), Vector2(1.0f, 1.0f), rand);
	IntersectionInfo info;
	info.pos = r.orig;
	info.normal = camera->get_dir();
	info.triangle_id = -1;
	result.push_back(Vertex(info, BSDF()));
	trace(result, r, 1, max_eye_events, rand);
	return result;
}

Path BidirectionalRenderer::trace_light_path(StateSequence &rand) {
	Path result;
	result.reserve(max_light_events);
	if (max_light_events == 0) {
		return result;
	}
	real pdf;
	const Triangle &tri = scene->sample_triangle_light_emission(rand(), pdf);
	Vector3 pos = tri.sample_point(rand(), rand());
	Vector3 dir = random_diffuse(tri.normal, rand(), rand());
	Ray ray(pos + eps * dir, dir, 0);

	IntersectionInfo info;
	info.pos = pos;
	info.normal = tri.normal;
	info.geometry_normal = tri.normal;
	info.triangle_id = tri.id;

	Vector3 u = normalized(tri.v[1] - tri.v[0]);
	Vector3 v = cross(tri.normal, u);
	info.to_world = Matrix3(u, v, tri.normal);
	info.to_local = glm::transpose(info.to_world);

	BSDF bsdf(scene, info);
	Vertex vertex(info, bsdf);
	vertex.event = (int)SurfaceScatteringFlags::emit;
	vertex.pdf = glm::dot(info.normal, dir) / pi;
	result.push_back(vertex);
	trace(result, ray, 1, max_light_events, rand);
	return result;
}

bool BidirectionalRenderer::connectable(int num_eye_vertices, int num_light_vertices,
	const Vertex &eye_end, const Vertex &light_end) {
	const Vector3 dir = normalize(light_end.pos - eye_end.pos);
	if ((num_eye_vertices == 1) && (num_light_vertices >= 1)) {
		// Light tracing
		if (SurfaceEventClassifier::is_delta(light_end.event) || dot(light_end.normal, dir) > -eps ||
			dot(camera->get_dir(), dir) < eps) {
			return false;
		}
	}
	else {
		// Otherwise, vertex connection
		if (!SurfaceEventClassifier::is_delta(light_end.event) && !SurfaceEventClassifier::is_delta(eye_end.event)) {
			if (light_end.triangle_id == eye_end.triangle_id || // on the same triangle?
				(dot(dir, light_end.normal) > -eps && num_light_vertices == 1)) {
				return false;
			}
		}
		else {
			return false;
		}
	}
	Ray r(eye_end.pos, dir);
	int tri_id = sg->query_hit_triangle_id(r);
	return tri_id == light_end.triangle_id; // occluded?
}

double BidirectionalRenderer::path_pdf(const Path &path,
	const int num_eye_vert_spec, const int num_light_vert_spec) {
	int path_length = (int)path.size() - 1;
	double p = 1.0;
	const int num_eye_vertices = num_eye_vert_spec;
	const int num_light_vertices = num_light_vert_spec;
	const int is_vm = ((int)path.size() - 1 == num_eye_vertices + num_light_vertices - 2);
	for (int i = -1; i <= num_eye_vertices - 2; i++) {
		if (i == -1) {
			p = p * camera->get_pixel_scaling();
		}
		else if (i == 0) {
			Vector3 d0 = normalize(path[1].pos - path[0].pos);
			double c = dot(d0, camera->get_dir());
			double distance_to_screen = 1.0f / c;
			distance_to_screen = distance_to_screen * distance_to_screen;
			p = p / (c / distance_to_screen);
			// NOTE: above....
			p = p * direction_to_area(path[0], path[1]);
			p = p / camera->get_pixel_scaling();
		}
		else {
			Vector3 in_dir = normalize(path[i - 1].pos - path[i].pos);
			Vector3 out_dir = normalize(path[i + 1].pos - path[i].pos);
			if (path[i].connected) {
				p = p * path[i].bsdf.probability_density(in_dir, out_dir);
			}
			else {
				p = p * path[i].pdf;
			}
			p = p * direction_to_area(path[i], path[i + 1]);
		}
	}
	if (p == 0.0) return p; // Shortcut
	for (int i = -1; i <= num_light_vertices - 2 - is_vm; i++) {
		if (i == -1) {
			// Light sample PDF
			int id = path[path_length].triangle_id;
			p = p * scene->get_triangle_pdf(id) / scene->get_triangle(id).area;
		}
		else if (i == 0) {
			Vector3 in_dir = normalize(
				path[path_length - 1].pos - path[path_length].pos);
			p = p * dot(path[path_length].normal, in_dir) / pi;
			p = p * direction_to_area(path[path_length], path[path_length - 1]);
		}
		else {
			if (path[path_length - i].connected) {
				Vector3 in_dir = normalize(
					path[path_length - (i - 1)].pos - path[path_length - i].pos);
				Vector3 out_dir = normalize(
					path[path_length - (i + 1)].pos - path[path_length - i].pos);
				p = p * path[path_length - i].bsdf.probability_density(in_dir, out_dir);
			}
			else {
				p = p * path[path_length - i].pdf;
			}
			p = p * direction_to_area(path[path_length - i], path[path_length - (i + 1)]);
		}
	}
	if (p == 0.0) return p; // Shortcut
	if (is_vm) {
		const Vertex &light_end = path[num_eye_vertices];
		const Vertex &eye_end = path[num_eye_vertices - 1];
		if (SurfaceEventClassifier::is_delta(eye_end.event)) p = 0;
		else {
			p *= light_end.pdf * direction_to_area(light_end, eye_end) * vm_pdf_constant;
		}
	}
	return p;
}

double BidirectionalRenderer::path_total_pdf(const Path &path,
	bool including_connection, int merging_factor) {
	int path_length = (int)path.size() - 1;
	double vc_pdf(0), vm_pdf(0);
	// We have to calculate all the possibilities...
	// Part I: Vertex Connection
	if (including_connection) {
		// num_eye_vertices starts from 1, since we use pinhole camera here...
		for (int num_eye_vertices = 1; num_eye_vertices <= path_length + 1; num_eye_vertices++) {
			int num_light_vertices = (path_length + 1) - num_eye_vertices;
			if (num_eye_vertices > max_eye_events || num_light_vertices > max_light_events) {
				continue;
			}
			if (num_eye_vertices >= 2 && SurfaceEventClassifier::is_delta(path[num_eye_vertices - 1].event)) {
				continue;
			}
			if (num_light_vertices >= 2 && SurfaceEventClassifier::is_delta(path[num_eye_vertices].event)) {
				continue;
			}
			vc_pdf += path_pdf(path, num_eye_vertices, num_light_vertices);
		}
	}
	// Part II: Vertex Merging
	if (merging_factor > 0) {
		// For VM, we start from num_eye_vertices = 2
		for (int num_eye_vertices = 2; num_eye_vertices <= path_length; num_eye_vertices++) {
			// Ensure that num_light_vertices >= 2
			int num_light_vertices = (path_length + 2) - num_eye_vertices;
			if (num_eye_vertices > max_eye_events || num_light_vertices > max_light_events) {
				continue;
			}
			// Merging vertex can not be delta
			if (SurfaceEventClassifier::is_delta(path[num_eye_vertices - 1].event)) {
				continue;
			}
			vm_pdf += path_pdf(path, num_eye_vertices, num_light_vertices);
		}
	}
	return vc_pdf + vm_pdf * merging_factor;
}

PathContribution BidirectionalRenderer::connect(const Path &eye_path, const Path &light_path,
	const int num_eye_vert_spec,
	const int num_light_vert_spec, const int merging_factor) {
	PathContribution result;
	bool specified = (num_eye_vert_spec != -1) && (num_light_vert_spec != -1);

	for (int path_length = min_path_length; path_length <= max_path_length; path_length++) {
		Path full_path;
		full_path.resize(path_length + 1);
		for (int num_eye_vertices = 1; num_eye_vertices <= path_length + 1; num_eye_vertices++) {
			const int num_light_vertices = (path_length + 1) - num_eye_vertices;
			if (num_eye_vertices > (int)eye_path.size()) continue;
			if (num_light_vertices > (int)light_path.size()) continue;
			if (specified && ((num_eye_vert_spec != num_eye_vertices) ||
				(num_light_vert_spec != num_light_vertices)))
				continue;
			if (num_eye_vertices == 0) {
				continue;
			}
			else if (num_light_vertices == 0) {
				const Vertex &eye_end = eye_path[num_eye_vertices - 1];
				bool valid = scene->get_triangle_emission(eye_end.triangle_id) > 0 && eye_end.front;
				if (!valid) {
					continue;
				}
			}
			else {
				const Vertex &eye_end = eye_path[num_eye_vertices - 1];
				const Vertex &light_end = light_path[num_light_vertices - 1];
				if (!connectable(num_eye_vertices, num_light_vertices, eye_end, light_end)) {
					continue;
				}
			}
			for (int i = 0; i < num_eye_vertices; i++) full_path[i] = eye_path[i];
			for (int i = 0; i < num_light_vertices; i++) full_path[path_length - i] = light_path[i];
			real px, py;
			camera->get_pixel_coordinate(normalized(full_path[1].pos - full_path[0].pos), px, py);
			if (px < 0 || px > 1 || py < 0 || py > 1) {
				continue;
			}
			if (num_eye_vertices > 0) {
				full_path[num_eye_vertices - 1].connected = true;
			}
			if (num_light_vertices > 0) {
				full_path[num_eye_vertices].connected = true;
			}
			Vector3d f = path_throughput(full_path);
			if (max_component(f) <= 0.0f) {
				//printf("f\n");
				continue;
			}
			double p = path_pdf(full_path, num_eye_vertices, num_light_vertices);
			if (p <= 0.0f) {
				//printf("p\n");
				continue;
			}
			double w = mis_weight(full_path, num_eye_vertices, num_light_vertices, true, merging_factor);
			if (w <= 0.0f) {
				//printf("w\n");
				continue;
			}

			Vector3d c = f * double(w / p);
			if (print_path_policy == "all" ||
				(print_path_policy == "bright" && max_component(c) > luminance_clamping)) {
				printf("Abnormal Path: #Eye %d, #Light %d", num_eye_vertices, num_light_vertices);
				printf("  f = %.10f %.10f %.10f, p = %.10f, c = %.10f, %.10f, %.10f\n", f[0], f[1], f[2], p, c[0],
					c[1], c[2]);
				for (int i = 0; i <= path_length; i++) {
					auto &v = full_path[i];
					printf("  pos = %f %f %f, normal = %f %f %f\n", v.pos[0], v.pos[1], v.pos[2], v.normal[0],
						v.normal[1], v.normal[2]);
					auto &b = full_path[i].bsdf;
					if (i >= 1 && i < path_length) {
						Vector3 in = normalized(full_path[i - 1].pos - full_path[i].pos);
						Vector3 out = normalized(full_path[i + 1].pos - full_path[i].pos);
						auto p = b.probability_density(in, out);
						auto brdf = b.evaluate(in, out);
						printf("  #brdf = %s, pdf = %f, evaluate_bsdf = %f %f %f\n", b.get_name().c_str(), p,
							brdf.x, brdf.y,
							brdf.z);
					}
					else if (i == path_length) {
						Vector3 in = normalized(full_path[i - 1].pos - full_path[i].pos);
						Vector3 out = normalized(full_path[i].normal);
						auto p = b.probability_density(in, out);
						auto brdf = b.evaluate(in, out);
						printf("  #light brdf = %s, pdf = %f, evaluate_bsdf = %f %f %f\n", b.get_name().c_str(), p,
							brdf.x,
							brdf.y, brdf.z);
					}
				}
				printf("\n");
			}
			if (print_path_policy != "none" && (abnormal(f) || abnormal(p) || abnormal(c))) {
				printf("%d - %d\n", num_eye_vertices, num_light_vertices);
				printf("f = %.10f %.10f %.10f, p = %.10f, c = %.10f, %.10f, %.10f\n", f[0], f[1], f[2], p,
					c[0], c[1], c[2]);
				printf("Abnormal Path: #Eye %d, #Light %d", num_eye_vertices, num_light_vertices);
				printf("  f = %.10f %.10f %.10f, p = %.10f, c = %.10f, %.10f, %.10f\n", f[0], f[1], f[2], p, c[0],
					c[1], c[2]);
				for (int i = 0; i <= path_length; i++) {
					auto &v = full_path[i];
					printf("  pos = %f %f %f, normal = %f %f %f\n", v.pos[0], v.pos[1], v.pos[2], v.normal[0],
						v.normal[1], v.normal[2]);
					auto &b = full_path[i].bsdf;
					if (i >= 1 && i < path_length) {
						Vector3 in = normalized(full_path[i - 1].pos - full_path[i].pos);
						Vector3 out = normalized(full_path[i + 1].pos - full_path[i].pos);
						auto p = b.probability_density(in, out);
						auto brdf = b.evaluate(in, out);
						printf("  #brdf = %s, pdf = %f, evaluate_bsdf = %f %f %f\n", b.get_name().c_str(), p,
							brdf.x, brdf.y,
							brdf.z);
					}
					else if (i == path_length) {
						Vector3 in = normalized(full_path[i - 1].pos - full_path[i].pos);
						Vector3 out = normalized(full_path[i].normal);
						auto p = b.probability_density(in, out);
						auto brdf = b.evaluate(in, out);
						printf("  #light brdf = %s, pdf = %f, evaluate_bsdf = %f %f %f\n", b.get_name().c_str(), p,
							brdf.x,
							brdf.y, brdf.z);
					}
				}
				printf("\n");
				continue;
			}
			if (max_component(c) <= 0.0) continue;
			//printf("%d - %d\n", num_eye_vertices, num_light_vertices);
			result.push_back(Contribution(px, py, path_length, c));

			if (specified && (num_eye_vert_spec == num_eye_vertices) &&
				(num_light_vert_spec == num_light_vertices))
				return result;
		}
	}
	return result;
}

double
BidirectionalRenderer::mis_weight(const Path &path, const int num_eye_vert_spec, const int num_light_vert_spec,
	bool including_connection, int merging_factor) {
	const double p_i = path_pdf(path, num_eye_vert_spec, num_light_vert_spec); // TODO: not necessary re-evaluation
	const double p_all = path_total_pdf(path, including_connection, merging_factor);
	if ((p_i == 0.0) || (p_all == 0.0)) {
		return 0.0;
	}
	else {
		return std::max(std::min(p_i / p_all, 1.0), 0.0);
	}
}

Vector3d BidirectionalRenderer::path_throughput(const Path &path) {
	Vector3d f(1.0f);
	for (int i = 0; i < path.size(); i++) {
		if (i == 0) {
			// Tricky camera throughput...
			Vector3d d0 = path[1].pos - path[0].pos;
			const double dist2 = dot(d0, d0);
			d0 = d0 * double(1.0 / sqrt(dist2));
			const double c = dot(d0, camera->get_dir());
			const double ds2 = 1.0 / (c * c);
			f = f * double(fabs(dot(d0, path[1].normal) / dist2 / c * ds2));
		}
		else if (i == ((int)path.size() - 1)) {
			if (path[i].bsdf.is_emissive()) {
				const Vector3 out_dir = normalize(path[i - 1].pos - path[i].pos);
				f = f * (Vector3d)path[i].bsdf.evaluate(path[i].normal, out_dir);
			}
			else {
				// Last event must be emission
				f *= 0;
			}
		}
		else {
			// No emissive material in the middle.
			const Vector3 in_dir = !path[i].connected ? path[i].in_dir : normalize(path[i - 1].pos - path[i].pos);
			const Vector3 out_dir = !path[i].connected ? path[i].out_dir : normalize(path[i + 1].pos - path[i].pos);
			if (path[i].bsdf.is_emissive() ||
				abs(dot(in_dir, path[i].normal)) < eps ||
				abs(dot(out_dir, path[i].normal)) < eps) {
				f *= 0.0f;
				return f;
			}
			Vector3d bsdf;
			if (path[i].connected) {
				// For end points of eye/light path, we need to re-evaluate bsdf
				bsdf = path[i].bsdf.evaluate(in_dir, out_dir);
			}
			else {
				bsdf = path[i].f;
			}
			f *= bsdf * geometry_term(path[i], path[i + 1]);
		}
		if (max_component(f) == 0.0f) return f;
	}
	return f;
}

double BidirectionalRenderer::geometry_term(const Vertex &current, const Vertex &next) {
	const Vector3 v = next.pos - current.pos;
	const real dist2 = dot(v, v);
	return fabs(dot(current.normal, v) * dot(next.normal, v)) / (dist2 * dist2);
}

double BidirectionalRenderer::direction_to_area(const Vertex &current, const Vertex &next) {
	const Vector3 v = next.pos - current.pos;
	const real dist2 = dot(v, v);
	return fabs(dot(next.normal, v)) / (dist2 * sqrt(dist2));
}

void BidirectionalRenderer::write_path_contribution(const PathContribution &pc, const real scaling) {
	real total_scaling = scaling * pc.get_scaling();
	for (auto &cont : pc.contributions) {
		if (0 <= cont.x && cont.x <= 1 - eps && 0 <= cont.y && cont.y <= 1 - eps) {
			if (abnormal(cont.c)) {
				P(cont.c);
				continue;
			}
			if (luminance_clamping > 0 && max_component(cont.c) > luminance_clamping) {
				P(cont.c);
				continue;
			}
			int ix = (int)floor(cont.x * width), iy = (int)floor(cont.y * height);
			output_lock.lock();
			this->buffer[ix][iy] += width * height * total_scaling * cont.c;
			output_lock.unlock();
		}
	}
}

TC_NAMESPACE_END

