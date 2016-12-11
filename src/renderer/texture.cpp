#include "texture.h"

TC_NAMESPACE_BEGIN

    TC_INTERFACE_DEF(Texture, "texture");

    class ConstantTexture final : public Texture {
    private:
        Vector3 val;
    public:
        void initialize(const Config &config) override {
            val = config.get_vec3("value");
        }
        virtual Vector3 sample(const Vector3 &coord) const override {
            return val;
        }
    };

    class ImageTexture : public Texture {
    protected:
        ImageBuffer<Vector3> image;
    public:
        void initialize(const Config &config) override {
            image.load(config.get_string("filename"));
        }
        virtual Vector3 sample(const Vector3 &coord_) const override  {
			Vector2 coord(coord_.x - floor(coord_.x), coord_.y - floor(coord_.y));
            return image.sample_relative_coord(coord);
        }
    };
    class TaichiTexture : public Texture {
	protected:
		real scale;
		real rotation_c, rotation_s;
    public:
        void initialize(const Config &config) override {
			scale = config.get("scale", 1.0f);
			real rotation = config.get("rotation", 0.0f);
			rotation_c = std::cos(rotation);
			rotation_s = std::sin(rotation);
        }
		static bool inside(Vector2 p, Vector2 c, real r) {
			return (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) <= r * r;
		}
		static bool inside_left(Vector2 p, Vector2 c, real r) {
			return inside(p, c, r) && p.x < c.x;
		}
		static bool inside_right(Vector2 p, Vector2 c, real r) {
			return inside(p, c, r) && p.x >= c.x;
		}
		bool is_white(Vector2 p_) const {
			p_ -= Vector2(0.5f);
			Vector2 p(p_.x * rotation_c + p_.y * rotation_s, -p_.x * rotation_s + p_.y * rotation_c);
			p += Vector2(0.5f);
			if (!inside(p, Vector2(0.50f, 0.50f), 0.5f)) {
				return true;
			}
			if (!inside(p, Vector2(0.50f, 0.50f), 0.5f * scale)) {
				return false;
			}
			p = Vector2(0.5f) + (p - Vector2(0.5f)) * (1.0f / scale);
			if (inside(p, Vector2(0.50f, 0.25f), 0.08f)) {
				return true;
			}
			if (inside(p, Vector2(0.50f, 0.75f), 0.08f)) {
				return false;
			}
			if (inside(p, Vector2(0.50f, 0.25f), 0.25f)) {
				return false;
			}
			if (inside(p, Vector2(0.50f, 0.75f), 0.25f)) {
				return true;
			}
			if (p.x < 0.5f) {
				return true;
			}
			else {
				return false;
			}
		}
        virtual Vector3 sample(const Vector2 &coord) const override {
			return Vector3(is_white(coord) ? 1.0f : 0.0f);
        }
        virtual Vector3 sample(const Vector3 &coord) const override {
            return sample(Vector2(coord.x, coord.y));
        }
    };
	class LinearOpTexture: public Texture {
	protected:
		real alpha, beta;
		bool need_clamp;
		std::shared_ptr<Texture> tex1, tex2;
	public:
        void initialize(const Config &config) override {
			alpha = config.get_real("alpha");
			beta = config.get_real("beta");
			need_clamp = config.get("need_clamp", false);
			tex1 = AssetManager::get_asset<Texture>(config.get_int("tex1"));
			tex2 = AssetManager::get_asset<Texture>(config.get_int("tex2"));
        }
        virtual Vector3 sample(const Vector3 &coord) const override  {
			auto p = alpha * tex1->sample(coord) + beta * tex2->sample(coord);
			if (need_clamp) {
				for (int i = 0; i < 3; i++) {
					p[i] = clamp(p[i], 0.0f, 1.0f);
				}
			}
			return p;
        }
	};
	class RasterizedTexture: public Texture {
	protected:
		ImageBuffer<Vector3> cache;
		int resolution_x;
		int resolution_y;
	public:
        void initialize(const Config &config) override {
			auto tex = AssetManager::get_asset<Texture>(config.get_int("tex"));
			resolution_x = config.get_int("resolution_x");
			resolution_y = config.get_int("resolution_y");
			cache = ImageBuffer<Vector3>(resolution_x, resolution_y);
			for (int i = 0; i < resolution_x; i++) {
				for (int j = 0; j < resolution_y; j++) {
					cache.set(i, j, tex->sample(
						Vector2((i + 0.5f) / resolution_x, (j + 0.5f) / resolution_y)));
				}
			}
        }
        virtual Vector3 sample(const Vector2 &coord) const override  {
			return cache.sample_relative_coord(coord);
        }
		virtual Vector3 sample(const Vector3 &coord) const override  {
			return cache.sample_relative_coord(Vector2(coord.x, coord.y));
		}
	};
	class MultiplicationTexture: public Texture {
	protected:
		std::shared_ptr<Texture> tex1;
		std::shared_ptr<Texture> tex2;
	public:
        void initialize(const Config &config) override {
			tex1 = AssetManager::get_asset<Texture>(config.get_int("tex1"));
			tex2 = AssetManager::get_asset<Texture>(config.get_int("tex2"));
        }
        virtual Vector3 sample(const Vector3 &coord) const override  {
			return tex1->sample(coord) * tex2->sample(coord);
        }
	};
	class CheckerboardTexture: public Texture {
	protected:
		real repeat_u, repeat_v, repeat_p;
		real inv_repeat_u, inv_repeat_v, inv_repeat_p;
		std::shared_ptr<Texture> tex1, tex2;
	public:
        void initialize(const Config &config) override {
			tex1 = AssetManager::get_asset<Texture>(config.get_int("tex1"));
			tex2 = AssetManager::get_asset<Texture>(config.get_int("tex2"));
			repeat_u = config.get_real("repeat_u");
			repeat_v = config.get_real("repeat_v");
			repeat_p = config.get("repeat_p", 1.0f);
        }
        virtual Vector3 sample(const Vector3 &coord) const override  {
			int p = (int)floor(coord.x * repeat_u), q = (int)floor(coord.y * repeat_v),
					r = (int)floor(coord.z * repeat_p);
			return ((p + q + r) % 2 == 0 ? tex1 : tex2)->sample(coord);
        }
	};
	class RepeaterTexture: public Texture {
	protected:
		real repeat_u, repeat_v, repeat_p;
		real inv_repeat_u, inv_repeat_v, inv_repeat_p;
		std::shared_ptr<Texture> tex;
	public:
        void initialize(const Config &config) override {
			repeat_u = config.get_real("repeat_u");
			repeat_v = config.get_real("repeat_v");
			repeat_p = config.get("repeat_p", 1.0f);
			inv_repeat_u = 1.0f / repeat_u;
			inv_repeat_v = 1.0f / repeat_v;
			inv_repeat_p = 1.0f / repeat_p;
			tex = AssetManager::get_asset<Texture>(config.get_int("tex"));
        }
        virtual Vector3 sample(const Vector3 &coord) const override  {
			real u = coord.x - floor(coord.x * repeat_u) * inv_repeat_u;
			real v = coord.y - floor(coord.y * repeat_v) * inv_repeat_v;
			real p = coord.y - floor(coord.z * repeat_p) * inv_repeat_p;
			return tex->sample(Vector3(u * repeat_u, v * repeat_v, p * repeat_p));
        }
	};
	class RotatedTexture: public Texture {
	protected:
		std::shared_ptr<Texture> tex;
		int times;
	public:
        void initialize(const Config &config) override {
			tex = AssetManager::get_asset<Texture>(config.get_int("tex"));
			times = config.get_int("times");
        }
        virtual Vector3 sample(const Vector3 &coord_) const override  {
			auto coord = coord_;
			for (int i = 0; i < times; i++) {
				coord = Vector3(-coord.x, coord.y, coord.z);
			}
			return tex->sample(coord);
        }
	};
	class FlippedTexture: public Texture {
	protected:
		std::shared_ptr<Texture> tex;
		int flip_axis;
	public:
        void initialize(const Config &config) override {
			tex = AssetManager::get_asset<Texture>(config.get_int("tex"));
			flip_axis = config.get_int("flip_axis");
        }
        virtual Vector3 sample(const Vector3 &coord_) const override {
			auto coord = coord_;
			coord[flip_axis] = 1.0f - coord[flip_axis];
			return tex->sample(coord);
        }
	};
	class BoundedTexture: public Texture {
	protected:
		std::shared_ptr<Texture> tex;
		int bound_axis;
		Vector2 bounds;
		Vector3 outside_val;
	public:
        void initialize(const Config &config) override {
			tex = AssetManager::get_asset<Texture>(config.get_int("tex"));
			bound_axis = config.get_int("axis");
			bounds = config.get_vec2("bounds");
			outside_val = config.get_vec3("outside_val");
        }
        virtual Vector3 sample(const Vector3 &coord_) const override {
			auto coord = coord_;
			if (bounds[0] <= coord[bound_axis] && coord[bound_axis] < bounds[1])
				return tex->sample(coord);
			else
				return outside_val;
        }
	};
    TC_IMPLEMENTATION(Texture, ConstantTexture, "const");
    TC_IMPLEMENTATION(Texture, ImageTexture, "image");
    TC_IMPLEMENTATION(Texture, TaichiTexture, "taichi");
    TC_IMPLEMENTATION(Texture, LinearOpTexture, "linear_op");
    TC_IMPLEMENTATION(Texture, RasterizedTexture, "rasterize");
    TC_IMPLEMENTATION(Texture, MultiplicationTexture, "mul");
    TC_IMPLEMENTATION(Texture, CheckerboardTexture, "checkerboard");
    TC_IMPLEMENTATION(Texture, RepeaterTexture, "repeater");
    TC_IMPLEMENTATION(Texture, RotatedTexture, "rotated");
    TC_IMPLEMENTATION(Texture, FlippedTexture, "flipped");
    TC_IMPLEMENTATION(Texture, BoundedTexture, "bounded");
TC_NAMESPACE_END

