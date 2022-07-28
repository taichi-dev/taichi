#include "taichi/common/logging.h"
#include "taichi/util/image_io.h"

#include "stb_image.h"
#include "stb_image_write.h"

namespace taichi {

void imwrite(const std::string &filename,
             size_t ptr,
             int resx,
             int resy,
             int comp) {
  void *data = (void *)ptr;
  TI_ASSERT_INFO(filename.size() >= 5, "Bad image file name");
  int result = 0;
  std::string suffix = filename.substr(filename.size() - 4);
  if (suffix == ".png") {
    result =
        stbi_write_png(filename.c_str(), resx, resy, comp, data, comp * resx);
  } else if (suffix == ".bmp") {
    result = stbi_write_bmp(filename.c_str(), resx, resy, comp, data);
  } else if (suffix == ".jpg") {
    result = stbi_write_jpg(filename.c_str(), resx, resy, comp, data, 95);
  } else {
    TI_ERROR("Unknown image file suffix {}", suffix);
  }

  if (!result) {
    TI_ERROR("Cannot write image file [{}]", filename);
  }
  TI_TRACE("saved image {}: {}x{}x{}", filename, resx, resy, comp);
}

std::vector<size_t> imread(const std::string &filename, int comp) {
  int resx = 0, resy = 0;
  void *data = stbi_load(filename.c_str(), &resx, &resy, &comp, comp);
  if (!data) {
    TI_ERROR("Cannot read image file [{}]", filename);
  }
  TI_TRACE("loaded image {}: {}x{}x{}", filename, resx, resy, comp);

  std::vector<size_t> ret = {(size_t)data, (size_t)resx, (size_t)resy,
                             (size_t)comp};
  return ret;
}

}  // namespace taichi
