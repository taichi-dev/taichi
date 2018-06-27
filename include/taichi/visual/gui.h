#pragma once

#include <taichi/taichi>
#include <numeric>

TC_NAMESPACE_BEGIN

#if defined(TC_PLATFORM_LINUX)

class Canvas {
 public:
  Array2D<Vector4> &img;
  Matrix3 transform_matrix;

  Canvas(Array2D<Vector4> &img) : img(img) {
    transform_matrix = Matrix3(Vector3(img.get_res().cast<real>(), 1.0_f));
  }

  TC_FORCE_INLINE Vector2 transform(Vector2 x) {
    return Vector2(transform_matrix * Vector3(x, 1.0_f));
  }

  void line(Vector2 start, Vector2 end, Vector4 color) {
    // convert to screen space
    start = transform(start);
    end = transform(end);
    real len = length(end - start);
    int samples = (int)len * 2 + 4;
    for (int i = 0; i < samples; i++) {
      real alpha = (1.0_f / (samples - 1)) * i;
      Vector2i coord = (lerp(alpha, start, end)).floor().cast<int>();
      if (img.inside(coord)) {
        img[coord] = color;
      }
    }
  }

  void triangle(Vector2 a, Vector2 b, Vector2 c, Vector4 color) {
    a = transform(a);
    b = transform(b);
    c = transform(c);

    // real points[3] = {a, b, c};
    // std::sort(points, points + 3, [](const Vector2 &a, const Vector2 &b) {
    //  return a.y < b.y;
    //});
    Vector2 limits[2];
    limits[0].x = min(a.x, min(b.x, c.x));
    limits[0].y = min(a.y, min(b.y, c.y));
    limits[1].x = max(a.x, max(b.x, c.x));
    limits[1].y = max(a.y, max(b.y, c.y));
    for (int i = std::floor(limits[0].x); i < std::ceil(limits[1].x); i++) {
      for (int j = std::floor(limits[0].y); j < std::ceil(limits[1].y); j++) {
        Vector2 pixel(i + 0.5_f, j + 0.5_f);
        bool inside_a = cross(pixel - a, b - a) <= 0;
        bool inside_b = cross(pixel - b, c - b) <= 0;
        bool inside_c = cross(pixel - c, a - c) <= 0;
        if (inside_a && inside_b && inside_c && img.inside(i, j)) {
          img[i][j] = color;
        }
      }
    }
  }

  void text(const std::string &str, real size, Vector4 color) {
    char *root_dir = std::getenv("TAICHI_REPO_DIR");
    TC_ASSERT(root_dir != nullptr);
    img.write_text(root_dir + std::string("/assets/fonts/go/Go-Bold.ttf"),
                   str, size, 0, 0, color);
  }

  void clear(Vector4 color) {
    img.reset(color);
  }
};

class CXImage;

class GUI {
 public:
  std::string window_name;
  int width, height;
  void *display;
  void *visual;
  unsigned long window;
  std::unique_ptr<CXImage> img;
  int frame_id = 0;
  const int fps = 60;
  float64 start_time;
  Array2D<Vector4> buffer;
  std::vector<real> last_frame_interval;
  std::unique_ptr<Canvas> canvas;
  float64 last_frame_time;
  bool key_pressed;

  void process_event();

  GUI(const std::string &window_name, int width = 800, int height = 800);

  Canvas &get_canvas() {
    return *canvas;
  }

  void update();

  void wait_key();

  ~GUI();
};

#endif

TC_NAMESPACE_END
