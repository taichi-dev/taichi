#include <taichi/io/image_reader.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(ImageReader, "image_reader");

class RawImageReader final : public ImageReader {
public:
	void initialize(const Config &config) override {
		
	}

	ImageBuffer<Vector4> read(const std::string &filepath) {
		
	}
};

TC_NAMESPACE_END