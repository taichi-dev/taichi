/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visualization/image_buffer.h>
#include <taichi/math/math.h>
#include <taichi/math/vector.h>

#if __GNUC__ == 5
#else
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image_write.h>

#define STB_TRUETYPE_IMPLEMENTATION

#include <stb_truetype.h>
#endif

TC_NAMESPACE_BEGIN

#define TC_GCC_5_ERROR \
  TC_ERROR("stb_image will trigger a bug of gcc-5; Recompile with gcc-6");

template <typename T>
void Array2D<T>::load(const std::string &filename, bool linearize) {
#if __GNUC__ == 5
  TC_GCC_5_ERROR
#else
  int channels;
  FILE *f = fopen(filename.c_str(), "rb");
  assert_info(f != nullptr, "Image file not found: " + filename);
  stbi_ldr_to_hdr_gamma(1.0_f);
  float32 *data =
      stbi_loadf(filename.c_str(), &this->res[0], &this->res[1], &channels, 0);
  assert_info(data != nullptr,
              "Image file load failed: " + filename +
                  " # Msg: " + std::string(stbi_failure_reason()));
  assert_info(channels == 1 || channels == 3 || channels == 4,
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
#endif
}

template <typename T>
void Array2D<T>::write(const std::string &filename) {
#if __GNUC__ == 5
  TC_GCC_5_ERROR
#else
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
  int write_result =
      stbi_write_png(filename.c_str(), this->res[0], this->res[1], comp,
                     &data[0], comp * this->res[0]);
  // assert_info((bool)write_result, "Can not write image file");
#endif
}

template <typename T>
void Array2D<T>::write_text(const std::string &font_fn,
                            const std::string &content_,
                            real size,
                            int dx,
                            int dy) {
#if __GNUC__ == 5
  TC_GCC_5_ERROR
#else
  std::vector<unsigned char> buffer(24 << 20, (unsigned char)0);
  std::vector<unsigned char> screen_buffer(
      (size_t)(this->res[0] * this->res[1]), (unsigned char)0);

  static stbtt_fontinfo font;
  int i, j, ascent, baseline, ch = 0;
  float xpos = 2;  // leave a little padding in case the character extends left
  static float scale;
  // TODO: cache loaded fonts?
  FILE *font_file = fopen(font_fn.c_str(), "rb");
  assert_info(font_file != nullptr,
              "Font file not found: " + std::string(font_fn));
  fread(&buffer[0], 1, 24 << 20, font_file);
  stbtt_InitFont(&font, &buffer[0], 0);

  scale = stbtt_ScaleForPixelHeight(&font, size);

  stbtt_GetFontVMetrics(&font, &ascent, 0, 0);
  baseline = (int)(ascent * scale);
  const std::string c_content = content_;
  const char *content = c_content.c_str();
  while (content[ch]) {
    int advance, lsb, x0, y0, x1, y1;
    float x_shift = xpos - (float)floor(xpos);
    stbtt_GetCodepointHMetrics(&font, content[ch], &advance, &lsb);
    stbtt_GetCodepointBitmapBoxSubpixel(&font, content[ch], scale, scale,
                                        x_shift, 0, &x0, &y0, &x1, &y1);
    stbtt_MakeCodepointBitmapSubpixel(
        &font,
        &screen_buffer[0] + this->res[0] * (baseline + y0) + (int)xpos + x0,
        x1 - x0, y1 - y0, 200, scale, scale, x_shift, 0, content[ch]);
    // note that this stomps the old data, so where character boxes overlap
    // (e.g. 'lj') it's wrong
    // because this API is really for baking character bitmaps into textures. if
    // you want to render
    // a sequence of characters, you really need to render each bitmap to a temp
    // buffer, then
    // "alpha blend" that into the working buffer
    xpos += (advance * scale);
    if (content[ch + 1])
      xpos += scale * stbtt_GetCodepointKernAdvance(&font, content[ch],
                                                    content[ch + 1]);
    ++ch;
  }
  if (dy < 0) {
    dy = this->res[1] + dy - 1;
  }
  for (j = 0; j < this->res[1]; ++j) {
    for (i = 0; i < this->res[0]; ++i) {
      int x = dx + i, y = dy + j;
      real alpha =
          screen_buffer[(this->res[1] - j - 1) * this->res[0] + i] / 255.0f;
      (*this)[x][y] = lerp(alpha, this->get(x, y), T(1.0_f));
    }
  }
#endif
}

template void Array2D<Vector3>::write_text(const std::string &font_fn,
                                           const std::string &content_,
                                           real size,
                                           int dx,
                                           int dy);

template void Array2D<Vector4>::write_text(const std::string &font_fn,
                                           const std::string &content_,
                                           real size,
                                           int dx,
                                           int dy);

template void Array2D<Vector3>::load(const std::string &filename, bool);

template void Array2D<Vector4>::load(const std::string &filename, bool);

template void Array2D<real>::write(const std::string &filename);

template void Array2D<Vector3>::write(const std::string &filename);

template void Array2D<Vector4>::write(const std::string &filename);

TC_NAMESPACE_END
