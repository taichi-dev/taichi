#include <taichi/visual/camera.h>

TC_NAMESPACE_BEGIN

class PinholeCamera : public Camera {
public:

	PinholeCamera() {}

	virtual void initialize(const Config &config) override;

	void initialize(Vector3 origin, Vector3 look_at, Vector3 up, int width, int height, real fov_angle,
		const Matrix4 &transform, real aspect_ratio);

	real get_pixel_scaling() override;

	virtual Ray sample(Vector2 offset, Vector2 size, real u, real v) override;

	void get_pixel_coordinate(Vector3 ray_dir, real &u, real &v) override;

private:
	real fov;
	real tan_half_fov;
	real aspect_ratio;
};

void PinholeCamera::initialize(const Config &config) {
	int width = config.get_int("width");
	int height = config.get_int("height");
	this->initialize(config.get_vec3("origin"), config.get_vec3("look_at"),
		config.get_vec3("up"), width, height, config.get_real("fov_angle"),
		Matrix4(1.0f), config.get("aspect_ratio", (real)width / height));
}

void PinholeCamera::initialize(Vector3 origin, Vector3 look_at, Vector3 up, int width, int height, real fov_angle,
	const Matrix4 &transform, real aspect_ratio) {
	fov = fov_angle / 180.0f * pi;
	this->origin = origin;
	this->look_at = look_at;
	this->up = up;
	this->width = width;
	this->height = height;
	set_dir_and_right();
	tan_half_fov = tan(fov / 2);
	this->aspect_ratio = aspect_ratio;
	this->transform = transform;
}

real PinholeCamera::get_pixel_scaling() {
	return sqr(tan_half_fov) * aspect_ratio;
}

Ray PinholeCamera::sample(Vector2 offset, Vector2 size, real u, real v) {
	Vector2 rand_offset = random_offset(offset, size, u, v);
	Vector3 local_dir = normalize(
		dir + rand_offset.x * tan_half_fov * right * aspect_ratio + rand_offset.y * tan_half_fov * up);
	Vector3 world_orig = multiply_matrix4(transform, origin, 1);
	Vector3 world_dir = normalized(multiply_matrix4(transform, local_dir, 0)); //TODO: why normalize here???
	return Ray(world_orig, world_dir, 0);
}

void PinholeCamera::get_pixel_coordinate(Vector3 ray_dir, real &u, real &v) {
	auto inv_transform = glm::inverse(transform);
	auto local_ray_dir = multiply_matrix4(inv_transform, ray_dir, 0);
	u = dot(local_ray_dir, right) / dot(local_ray_dir, dir) / tan_half_fov / aspect_ratio + 0.5f;
	v = dot(local_ray_dir, up) / dot(local_ray_dir, dir) / tan_half_fov + 0.5f;
}
TC_IMPLEMENTATION(Camera, PinholeCamera, "pinhole");

TC_NAMESPACE_END

