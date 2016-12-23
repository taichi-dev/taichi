#pragma once

#include <taichi/visual/camera.h>
#include <taichi/visual/scene.h>
#include <taichi/visual/scene_geometry.h>
#include <taichi/visualization/image_buffer.h>
#include <taichi/system/timer.h>
#include <taichi/common/meta.h>

TC_NAMESPACE_BEGIN
class Renderer : public Unit {
public:
	virtual void initialize(const Config &config);
	virtual void render_stage() {};
	virtual void set_scene(std::shared_ptr<Scene> scene);
	virtual ImageBuffer<Vector3> get_output() { return ImageBuffer<Vector3>(width, height); };
	virtual void write_output(std::string fn);

protected:
	std::shared_ptr<Camera> camera;
	std::shared_ptr<Scene> scene;
	std::shared_ptr<RayIntersection> ray_intersection;
	std::shared_ptr<SceneGeometry> sg;
	int width, height;
	int min_path_length, max_path_length;
	int num_threads;
	bool path_length_in_range(int path_length) {
		return min_path_length <= path_length && path_length <= max_path_length;
	}
};

TC_INTERFACE(Renderer);

class TemperatureRenderer : public Renderer {
public:
	void initialize(Config &config) {
		Renderer::initialize(config);
		buffer.initialize(width, height);
	}

	Vector3 trace(Ray &ray) {
		int tri_id = sg->query_hit_triangle_id(ray);
		real temp = 0;
		if (tri_id != -1) {
			temp = scene->triangles[tri_id].temperature;
		}
		return Vector3(temp);
	}

	void render_stage() override {
		int spp = 1;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				Vector2 offset(float(i) / width, float(j) / height);
				Vector2 size(1.0f / width, 1.0f / height);
				Vector3 sum(0);
				for (int k = 0; k < spp; k++) {
					//Ray ray = camera->sample(offset, size);
					//sum += trace(ray);
				}
				buffer[i][j] = 1.0f / spp * sum;
			}
		}
	}

	virtual ImageBuffer<Vector3> get_output() override {
		return buffer;
	}

private:
	ImageBuffer<Vector3> buffer;
};


TC_NAMESPACE_END

