/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/math/math.h"
#include "taichi/math/linalg.h"
#include "taichi/util/base64.h"

#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

TI_NAMESPACE_BEGIN

template <typename T>
void Array2D<T>::load_image(const std::string &filename, bool linearize) {
  int channels;
  FILE *f = fopen(filename.c_str(), "rb");
  TI_ASSERT_INFO(f != nullptr, "Image file not found: " + filename);
  stbi_ldr_to_hdr_gamma(1.0_f);
  float32 *data =
      stbi_loadf(filename.c_str(), &this->res[0], &this->res[1], &channels, 0);
  TI_ASSERT_INFO(data != nullptr,
                 "Image file load failed: " + filename +
                     " # Msg: " + std::string(stbi_failure_reason()));
  TI_ASSERT_INFO(channels == 1 || channels == 3 || channels == 4,
                 "Image must have channel 1, 3 or 4: " + filename);
  this->initialize(Vector2i(this->res[0], this->res[1]));

  for (int i = 0; i < this->res[0]; i++) {
    for (int j = 0; j < this->res[1]; j++) {
      float32 *pixel_ =
          data + ((this->res[1] - 1 - j) * this->res[0] + i) * channels;
      Vector4 pixel;
      if (channels == 1) {
        pixel = Vector4(pixel_[0]);
      } else {
        pixel = Vector4(pixel_[0], pixel_[1], pixel_[2], pixel_[3]);
      }
      if (linearize) {
        pixel = pixel.pow(2.2f);
      }
      (*this)[i][j][0] = pixel[0];
      (*this)[i][j][1] = pixel[1];
      (*this)[i][j][2] = pixel[2];
      if (channels == 4 && std::is_same<T, Vector4>::value)
        (*this)[i][j][3] = pixel[3];
    }
  }

  stbi_image_free(data);
}

template <typename T>
void Array2D<T>::write_as_image(const std::string &filename) {
  int comp = 3;
  std::vector<unsigned char> data(this->res[0] * this->res[1] * comp);
  for (int i = 0; i < this->res[0]; i++) {
    for (int j = 0; j < this->res[1]; j++) {
      for (int k = 0; k < comp; k++) {
        data[j * this->res[0] * comp + i * comp + k] =
            (unsigned char)(255.0f *
                            clamp(VectorND<3, real>(
                                      this->data[i * this->res[1] +
                                                 (this->res[1] - j - 1)])[k],
                                  0.0_f, 1.0_f));
      }
    }
  }
  TI_ASSERT(filename.size() >= 5);
  int write_result = 0;
  std::string suffix = filename.substr(filename.size() - 4);
  if (suffix == ".png") {
    write_result = stbi_write_png(filename.c_str(), this->res[0], this->res[1],
                                  comp, &data[0], comp * this->res[0]);
  } else if (suffix == ".bmp") {
    // TODO: test
    write_result = stbi_write_bmp(filename.c_str(), this->res[0], this->res[1],
                                  comp, &data[0]);
  } else if (suffix == ".jpg") {
    // TODO: test
    write_result = stbi_write_jpg(filename.c_str(), this->res[0], this->res[1],
                                  comp, &data[0], 95);
  } else {
    TI_ERROR("Unknown suffix {}", suffix);
  }

  TI_ASSERT_INFO((bool)write_result, "Cannot write image file");
}

std::map<std::string, stbtt_fontinfo> fonts;
std::map<std::string, std::vector<uint8>> font_buffers;

template <typename T>
void Array2D<T>::write_text(const std::string &font_fn,
                            const std::string &content_,
                            real size,
                            int dx,
                            int dy,
                            T color) {
  std::vector<unsigned char> screen_buffer(
      (size_t)(this->res[0] * this->res[1]), (unsigned char)0);

  int i, j, ascent, baseline, ch = 0;
  float xpos = 2;  // leave a little padding in case the character extends left

  stbtt_fontinfo font;
  if (fonts.find(font_fn) == fonts.end()) {
    auto buffer_size = 24 << 20;
    font_buffers[font_fn] =
        std::vector<unsigned char>(buffer_size, (unsigned char)0);
    if (font_fn != "") {
      FILE *font_file = fopen(font_fn.c_str(), "rb");
      TI_ASSERT_INFO(font_file != nullptr,
                     "Font file not found: " + std::string(font_fn));
      trash(fread(&font_buffers[font_fn][0], 1, buffer_size, font_file));
      fclose(font_file);
    } else {
      TI_NOT_IMPLEMENTED
    }
    stbtt_InitFont(&font, &font_buffers[font_fn][0], 0);
    fonts[font_fn] = font;
  } else {
    font = fonts[font_fn];
  }

  real scale = stbtt_ScaleForPixelHeight(&font, size);

  stbtt_GetFontVMetrics(&font, &ascent, nullptr, nullptr);
  baseline = (int)(ascent * scale);
  const std::string c_content = content_;
  const char *content = c_content.c_str();
  while (content[ch]) {
    int advance, lsb, x0, y0, x1, y1;
    float32 x_shift = xpos - (float32)floor(xpos);
    stbtt_GetCodepointHMetrics(&font, content[ch], &advance, &lsb);
    stbtt_GetCodepointBitmapBoxSubpixel(&font, content[ch], scale, scale,
                                        x_shift, 0, &x0, &y0, &x1, &y1);
    stbtt_MakeCodepointBitmapSubpixel(
        &font,
        &screen_buffer[0] + this->res[0] * (baseline + y0) + (int)xpos + x0,
        x1 - x0, y1 - y0, this->res[0], scale, scale, x_shift, 0, content[ch]);
    // note that this stomps the old data, so where character boxes overlap
    // (e.g. 'lj') it's wrong
    xpos += (advance * scale);
    if (content[ch + 1])
      xpos += scale * stbtt_GetCodepointKernAdvance(&font, content[ch],
                                                    content[ch + 1]);
    ++ch;
  }
  for (j = 0; j < this->res[1]; ++j) {
    for (i = 0; i < this->res[0]; ++i) {
      int x = dx + i, y = dy + j - this->res[1];
      auto index = ((this->res[1] - j - 1) * this->res[0] + i);
      real alpha = screen_buffer[index] / 255.0f;
      if (inside(x, y) && alpha != 0) {
        (*this)[x][y] = lerp(alpha, this->get(x, y), color);
      }
    }
  }
}

template void Array2D<Vector3>::write_text(const std::string &font_fn,
                                           const std::string &content_,
                                           real size,
                                           int dx,
                                           int dy,
                                           Vector3);

template void Array2D<Vector4>::write_text(const std::string &font_fn,
                                           const std::string &content_,
                                           real size,
                                           int dx,
                                           int dy,
                                           Vector4);

template void Array2D<Vector3f>::load_image(const std::string &filename, bool);

template void Array2D<Vector4f>::load_image(const std::string &filename, bool);

template void Array2D<Vector3d>::load_image(const std::string &filename, bool);

template void Array2D<Vector4d>::load_image(const std::string &filename, bool);

template void Array2D<float32>::write_as_image(const std::string &filename);

template void Array2D<float64>::write_as_image(const std::string &filename);

template void Array2D<Vector3f>::write_as_image(const std::string &filename);

template void Array2D<Vector4f>::write_as_image(const std::string &filename);

template void Array2D<Vector3d>::write_as_image(const std::string &filename);

template void Array2D<Vector4d>::write_as_image(const std::string &filename);

void write_pgm(Array2D<real> img, const std::string &fn) {
  std::ofstream fs(fn, std::ios_base::binary);
  Vector2i res = img.get_res();
  fs << fmt::format("P5\n{} {}\n{}\n", res[0], res[1], 255);
  for (int j = 0; j < res[1]; j++) {
    std::string line;
    for (int i = 0; i < res[0]; i++) {
      uint8_t v = clamp((int)(img[i][res[1] - j - 1] * 255), 0, 255);
      line.push_back(v);
    }
    fs.write(line.c_str(), line.size());
  }
}

TI_NAMESPACE_END
