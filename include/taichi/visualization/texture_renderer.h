/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>
#include <taichi/system/opengl.h>
#include <taichi/visualization/image_buffer.h>

TC_NAMESPACE_BEGIN

class TextureRenderer {
 private:
  static GLuint program, vbo;
  static bool shared_resources_initialized;
  GLuint vao;
  GLuint texture;
  int width, height;
  vector<unsigned char> image;
  std::shared_ptr<GLWindow> context;

 public:
  TextureRenderer(std::shared_ptr<GLWindow> window, int height, int width);

  void resize(int height, int width);

  void reset();

  void set_pixel(int x, int y, Vector4 color);

  template <typename T>
  void set_texture(Array2D<T> image);

  void render();

  ~TextureRenderer();

  static Vector4 to_vec4(real dat) { return dat; }
  static Vector4 to_vec4(unsigned char dat) { return dat / 255.0f; }
  static Vector4 to_vec4(Vector2 dat) { return Vector4(dat.x, dat.y, 0, 1); }
  static Vector4 to_vec4(Vector3 dat) {
    return Vector4(dat.x, dat.y, dat.z, 1);
  }
  static Vector4 to_vec4(Vector4 dat) { return dat; }
};

template <typename T>
inline void TextureRenderer::set_texture(Array2D<T> image) {
  // assert_info(image.get_width() == width && image.get_height() == height,
  // "Texture size mismatch!");
  resize(image.get_width(), image.get_height());
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      set_pixel(i, j, to_vec4(image[i][j]));
    }
  }
}

TC_NAMESPACE_END
