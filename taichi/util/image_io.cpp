#include <taichi/common/util.h>
#include <taichi/util/image_io.h>

#if !defined(TI_AMALGAMATED)
#define TI_IMAGE_IO
#endif

#if defined(TI_IMAGE_IO)
#include <stb_image.h>
#include <stb_image_write.h>
#endif

TI_NAMESPACE_BEGIN

void tc_imwrite(const std::string &filename, size_t ptr, int resx, int resy, int comp)
{
#if defined(TI_IMAGE_IO)
  void *data = (void *)ptr;
  // TODO(archibate): throw python exceptions instead of abort...
  TI_ASSERT_INFO(filename.size() >= 5, "Bad image file name");
  int write_result = 0;
  std::string suffix = filename.substr(filename.size() - 4);
  if (suffix == ".png") {
    write_result = stbi_write_png(filename.c_str(), resx, resy,
                                  comp, data, comp * resx);
  } else if (suffix == ".bmp") {
    // TODO: test
    write_result = stbi_write_bmp(filename.c_str(), resx, resy,
                                  comp, data);
  } else if (suffix == ".jpg") {
    // TODO: test
    write_result = stbi_write_jpg(filename.c_str(), resx, resy,
                                  comp, data, 95);
  } else {
    TI_ERROR("Unknown image file suffix {}", suffix);
  }

  TI_ASSERT_INFO((bool)write_result, "Cannot write image file");
#else
  TI_ERROR(
      "'imwrite' is not implemented. Append -DTI_IMAGE_IO to "
      "compiler options if you are using taichi.h.");
#endif
}

TI_NAMESPACE_END
