/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/math/math.h>
#include <taichi/visualization/texture_renderer.h>

#ifdef TC_USE_OPENGL

TC_NAMESPACE_BEGIN

GLuint TextureRenderer::program, TextureRenderer::vbo;
bool TextureRenderer::shared_resources_initialized = false;

TextureRenderer::TextureRenderer(std::shared_ptr<GLWindow> context,
                                 int width,
                                 int height)
    : width(width), height(height) {
  texture = unsigned(-1);
  this->context = context;
  auto _ = context->create_context_guard();
  CGL;
  reset();
  if (!shared_resources_initialized) {
    float vbo_data[]{-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1};

    program = load_program("texture", "texture");

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 48, vbo_data, GL_STATIC_DRAW);
    shared_resources_initialized = true;
  }

  this->width = -1;
  this->height = -1;
  resize(width, height);
  CGL;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, false, 8, 0);
  glBindVertexArray(0);
  CGL;
}

void TextureRenderer::resize(int width, int height) {
  if (this->width == width && this->height == height)
    return;
  this->width = width;
  this->height = height;
  image.resize(height * width * 4);

  auto _ = context->create_context_guard();
  if (texture != unsigned(-1)) {
    glDeleteTextures(1, &texture);
  }
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, nullptr);
}

void TextureRenderer::reset() {
  if (image.size())
    memset(&image[0], 0, sizeof(unsigned char) * image.size());
}

void TextureRenderer::set_pixel(int x, int y, vec4 color) {
  unsigned char *p = &image[0] + 4 * (y * width + x);
  p[0] = (unsigned char)(clamp(color.r, 0.0f, 1.0f) * 255.0);
  p[1] = (unsigned char)(clamp(color.g, 0.0f, 1.0f) * 255.0);
  p[2] = (unsigned char)(clamp(color.b, 0.0f, 1.0f) * 255.0);
  p[3] = (unsigned char)(clamp(color.a, 0.0f, 1.0f) * 255.0);
}

void TextureRenderer::render() {
  CGL;
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glUseProgram(program);
  glActiveTexture(GL_TEXTURE0);
  CGL;
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                  GL_UNSIGNED_BYTE, &image[0]);
  glGenerateMipmap(GL_TEXTURE_2D);
  glDisable(GL_DEPTH_TEST);
  CGL;
  glBindVertexArray(vao);
  CGL;
  glUniform1i(glGetUniformLocation(program, "tex"), 0);
  CGL;
  glDrawArrays(GL_TRIANGLES, 0, 12);
  CGL;
  glBindVertexArray(0);
  CGL;
}

TextureRenderer::~TextureRenderer() {}

TC_NAMESPACE_END

#endif
