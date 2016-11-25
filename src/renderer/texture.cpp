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
    TC_IMPLEMENTATION(Texture, ConstantTexture, "const");
    TC_IMPLEMENTATION(Texture, ImageTexture, "image");
TC_NAMESPACE_END

