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
        virtual Vector3 sample(const Vector2 &coord) const override {
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
        virtual Vector3 sample(const Vector2 &coord) const override  {
            return image.sample_relative_coord(coord);
        }
    };
    class TaichiTexture : public Texture {
	protected:
		real scale;
    public:
        void initialize(const Config &config) override {
			scale = config.get("scale", 1.0f);
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
		bool is_white(Vector2 p) const {
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
        virtual Vector3 sample(const Vector2 &coord) const override  {
			auto p = alpha * tex1->sample(coord) + beta * tex2->sample(coord);
			if (need_clamp) {
				for (int i = 0; i < 3; i++) {
					p[i] = clamp(p[i], 0.0f, 1.0f);
				}
			}
			return p;
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
        virtual Vector3 sample(const Vector2 &coord) const override  {
			return tex1->sample(coord) * tex2->sample(coord);
        }
	};
	class CheckerboardTexture: public Texture {
	protected:
		real repeat_u, repeat_v;
		real inv_repeat_u, inv_repeat_v;
		std::shared_ptr<Texture> tex1, tex2;
	public:
        void initialize(const Config &config) override {
			tex1 = AssetManager::get_asset<Texture>(config.get_int("tex1"));
			tex2 = AssetManager::get_asset<Texture>(config.get_int("tex2"));
			repeat_u = config.get_real("repeat_u");
			repeat_v = config.get_real("repeat_v");
			inv_repeat_u = 1.0f / repeat_u;
			inv_repeat_v = 1.0f / repeat_v;
        }
        virtual Vector3 sample(const Vector2 &coord) const override  {
			int p = (int)floor(coord.x / inv_repeat_u), q = (int)floor(coord.y / inv_repeat_v);
			return ((p + q) % 2 == 0 ? tex1 : tex2)->sample(coord);
        }
	};
	class RepeaterTexture: public Texture {
	protected:
		real repeat_u, repeat_v;
		real inv_repeat_u, inv_repeat_v;
		std::shared_ptr<Texture> tex;
	public:
        void initialize(const Config &config) override {
			repeat_u = config.get_real("repeat_u");
			repeat_v = config.get_real("repeat_v");
			inv_repeat_u = 1.0f / repeat_u;
			inv_repeat_v = 1.0f / repeat_v;
			tex = AssetManager::get_asset<Texture>(config.get_int("tex"));
        }
        virtual Vector3 sample(const Vector2 &coord) const override  {
			real u = coord.x - floor(coord.x / inv_repeat_u) * inv_repeat_u;
			real v = coord.y - floor(coord.y / inv_repeat_v) * inv_repeat_v;
			return tex->sample(Vector2(u * repeat_u, v * repeat_v));
        }
	};
    TC_IMPLEMENTATION(Texture, ConstantTexture, "const");
    TC_IMPLEMENTATION(Texture, ImageTexture, "image");
    TC_IMPLEMENTATION(Texture, TaichiTexture, "taichi");
    TC_IMPLEMENTATION(Texture, LinearOpTexture, "linear_op");
    TC_IMPLEMENTATION(Texture, MultiplicationTexture, "mul");
    TC_IMPLEMENTATION(Texture, CheckerboardTexture, "checkerboard");
    TC_IMPLEMENTATION(Texture, RepeaterTexture, "repeater");
TC_NAMESPACE_END

