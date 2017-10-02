/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/envmap.h>
#include <taichi/visual/texture.h>
#include <taichi/common/asset_manager.h>

TC_NAMESPACE_BEGIN

TC_IMPLEMENTATION(EnvironmentMap, EnvironmentMap, "base");

void EnvironmentMap::initialize(const Config &config) {
  set_transform(Matrix4(1.0_f));
  if (config.has_key("filepath")) {
    image =
        std::make_shared<Array2D<Vector3>>(config.get<std::string>("filepath"));
    res[0] = image->get_width();
    res[1] = image->get_height();
  } else {
    assert_info(config.has_key("texture"),
                "Either `filenpath` or `texture` should be specified.");
    res = config.get("res", Vector2i(1024, 512));
    image = std::make_shared<Array2D<Vector3>>(res);
    Texture *tex = config.get_asset<Texture>("texture").get();
    *image = tex->rasterize3(res);
  }
  /*
  for (int j = 0; j < height; j++) {
      // conversion
      auto scale = sin(pi * (0.5f + j) / height);
      for (int i = 0; i < width; i++) {
          (*image)[i][j] *= scale;
      }
  }
  */
  for (int j = 0; j < res[1] - j - 1; j++) {
    for (int i = 0; i < res[0]; i++)
      std::swap((*image)[i][j], (*image)[i][res[1] - j - 1]);
  }

  build_cdfs();
  /*
  P("test");
  P(uv_to_direction(Vector2(0.0_f, 0.0_f)));
  P(uv_to_direction(Vector2(0.0_f, 0.5f)));
  P(uv_to_direction(Vector2(0.0_f, 1.0_f)));
  P(uv_to_direction(Vector2(0.5f, 0.25f)));
  P(uv_to_direction(Vector2(0.5f, 0.5f)));
  P(uv_to_direction(Vector2(0.5f, 0.75f)));
  P(uv_to_direction(Vector2(1.0_f, 0.0_f)));
  P(uv_to_direction(Vector2(1.0_f, 0.5f)));
  P(uv_to_direction(Vector2(1.0_f, 1.0_f)));

  for (int i = 0; i < 10; i++) {
      real x = rand(), y = rand();
      auto uv = Vector2(x, y);
      P(uv);
      P(direction_to_uv(uv_to_direction(uv)));
  }

  for (int i = 0; i < 100; i++) {
      RandomStateSequence rand(create_instance<Sampler>("prand"), i);
      real pdf;
      Vector3 illum;
      Vector3 dir;
      dir = sample_direction(rand, pdf, illum);
      P(dir);
      P(pdf);
      P(illum);
      P(luminance(illum) / pdf);
  }
  */
}

real EnvironmentMap::pdf(const Vector3 &dir) const {
  Vector2 uv = direction_to_uv(dir);
  return luminance(image->sample(uv.x, uv.y)) / avg_illum * (1.0_f / 4 / pi);
}

Vector3 EnvironmentMap::sample_direction(StateSequence &rand,
                                         real &pdf,
                                         Vector3 &illum) const {
  Vector2 uv;
  real row_pdf, row_cdf;
  real col_pdf, col_cdf;
  real row_sample = rand();
  real col_sample = rand();
  int row = row_sampler.sample(row_sample, row_pdf, row_cdf);
  int col = col_samplers[row].sample(col_sample, col_pdf, col_cdf);
  real u = col + 0.5f;  // (col_sample - col_cdf) / col_pdf;
  real v = row + 0.5f;  // (row_sample - row_cdf) / row_pdf;
  uv.x = u / res[0];
  uv.y = v / res[1];
  illum = sample_illum(uv);
  pdf = row_pdf * col_pdf * res[0] * res[1] / sin(pi * (0.5f + row) / res[1]);
  // P(luminance(illum) / pdf);
  return uv_to_direction(uv);
}

void EnvironmentMap::build_cdfs() {
  std::vector<real> row_pdf;
  avg_illum = 0;
  real total_weight = 0.0_f;
  for (int j = 0; j < res[1]; j++) {
    std::vector<real> col_pdf;
    real scale = sin(pi * (0.5f + j) / res[1]);
    real total = 0.0_f;
    for (int i = 0; i < res[0]; i++) {
      real pdf =
          luminance(image->sample((i + 0.5f) / res[0], (j + 0.5f) / res[1]));
      avg_illum += pdf * scale;
      total_weight += scale;
      total += pdf;
      col_pdf.push_back(pdf);
    }
    col_samplers.push_back(DiscreteSampler(col_pdf, true));
    row_pdf.push_back(total * scale);
  }
  row_sampler.initialize(row_pdf);
  avg_illum /= total_weight;
}

TC_NAMESPACE_END
