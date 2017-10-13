/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "ray_intersection.h"

TC_NAMESPACE_BEGIN

class BruteForceRayIntersection : public RayIntersection {
 public:
  void clear() override;

  void build() override;

  void query(Ray &ray) override;

  void add_triangle(Triangle &triangle) override;

 private:
  std::vector<Triangle> triangles;

  // Inherited via RayIntersection
  virtual bool occlude(Ray &ray) override;
};

class EmbreeRayIntersection : public RayIntersection {
 public:
  void clear() override;

  void build() override;

  void query(Ray &ray) override;

  void add_triangle(Triangle &triangle) override;

  virtual bool occlude(Ray &ray) override;

 private:
  std::vector<Triangle> triangles;
  RTCDevice rtc_device;
  RTCScene rtc_scene;
  int geom_id;
};

void BruteForceRayIntersection::clear() {
  triangles.clear();
}

void BruteForceRayIntersection::build() {
}

void BruteForceRayIntersection::query(Ray &ray) {
  for (auto &triangle : triangles) {
    triangle.intersect(ray);
  }
}

void BruteForceRayIntersection::add_triangle(Triangle &triangle) {
  triangles.push_back(triangle);
}

bool BruteForceRayIntersection::occlude(Ray &ray) {
  return false;
}

void EmbreeRayIntersection::clear() {
  triangles.clear();
  rtcDeleteScene(rtc_scene);
  rtcDeleteDevice(rtc_device);
}

/* error reporting function */
void error_handler(const RTCError code, const char *str = nullptr) {
  if (code == RTC_NO_ERROR)
    return;

  printf("Embree: ");
  switch (code) {
    case RTC_UNKNOWN_ERROR:
      printf("RTC_UNKNOWN_ERROR");
      break;
    case RTC_INVALID_ARGUMENT:
      printf("RTC_INVALID_ARGUMENT");
      break;
    case RTC_INVALID_OPERATION:
      printf("RTC_INVALID_OPERATION");
      break;
    case RTC_OUT_OF_MEMORY:
      printf("RTC_OUT_OF_MEMORY");
      break;
    case RTC_UNSUPPORTED_CPU:
      printf("RTC_UNSUPPORTED_CPU");
      break;
    case RTC_CANCELLED:
      printf("RTC_CANCELLED");
      break;
    default:
      printf("invalid error code");
      break;
  }
  if (str) {
    printf(" (");
    while (*str)
      putchar(*str++);
    printf(")\n");
  }
  assert(false);
}

void EmbreeRayIntersection::build() {
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  int num_triangles = (int)triangles.size(), num_vertices = num_triangles * 3;
  RTCGeometryFlags geom_flags = RTC_GEOMETRY_STATIC;

  rtc_device = rtcNewDevice(NULL);

  error_handler(rtcDeviceGetError(rtc_device));
  rtcDeviceSetErrorFunction(rtc_device, error_handler);

  rtc_scene = rtcDeviceNewScene(rtc_device, RTC_SCENE_STATIC, RTC_INTERSECT1);
  geom_id =
      rtcNewTriangleMesh(rtc_scene, geom_flags, num_triangles, num_vertices, 1);

  struct RTCVertex {
    float x, y, z, a;
  };
  struct RTCTriangle {
    int v[3];
  };

  RTCVertex *vertices =
      (RTCVertex *)rtcMapBuffer(rtc_scene, geom_id, RTC_VERTEX_BUFFER);
  for (int i = 0; i < num_triangles; i++) {
    for (int k = 0; k < 3; k++) {
      *(Vector4 *)(&vertices[i * 3 + k]) = Vector4(triangles[i].v[k], 0);
    }
  }
  rtcUnmapBuffer(rtc_scene, geom_id, RTC_VERTEX_BUFFER);

  RTCTriangle *triangles =
      (RTCTriangle *)rtcMapBuffer(rtc_scene, geom_id, RTC_INDEX_BUFFER);
  for (int i = 0; i < num_triangles; i++) {
    for (int k = 0; k < 3; k++) {
      triangles[i].v[k] = i * 3 + k;
    }
  }
  rtcUnmapBuffer(rtc_scene, geom_id, RTC_INDEX_BUFFER);

  rtcCommit(rtc_scene);
  error_handler(rtcDeviceGetError(rtc_device));
}

void EmbreeRayIntersection::query(Ray &ray) {
  RTCRay rtc_ray;
  *(Vector3 *)rtc_ray.org = ray.orig;
  *(Vector3 *)rtc_ray.dir = ray.dir;
  rtc_ray.tnear = eps * 10;
  rtc_ray.tfar = Ray::DIST_INFINITE;
  rtc_ray.time = 0.0_f;
  rtc_ray.mask = -1;
  rtc_ray.geomID = RTC_INVALID_GEOMETRY_ID;
  rtc_ray.primID = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect(rtc_scene, rtc_ray);
  ray.u = rtc_ray.u;
  ray.v = rtc_ray.v;
  ray.dist = rtc_ray.tfar;
  ray.triangle_id = rtc_ray.primID;
  return;
  Vector3 normal = Vector3(rtc_ray.Ng[0], rtc_ray.Ng[1],
                           rtc_ray.Ng[2]);  // What the hell happened to Ng???
  normal /= normal.abs().max();
  ray.geometry_normal = normalize(normal);
}

void EmbreeRayIntersection::add_triangle(Triangle &triangle) {
  triangles.push_back(triangle);
}

bool EmbreeRayIntersection::occlude(Ray &ray) {
  assert(false);  // TODO
  RTCRay rtc_ray;
  *(Vector3 *)rtc_ray.org = ray.orig;
  *(Vector3 *)rtc_ray.dir = ray.dir;
  rtc_ray.tnear = eps * 10;
  rtc_ray.tfar = ray.dist;
  rtc_ray.time = 0.0_f;
  rtc_ray.mask = -1;
  rtc_ray.geomID = RTC_INVALID_GEOMETRY_ID;
  rtc_ray.primID = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect(rtc_scene, rtc_ray);
  ray.u = rtc_ray.u;
  ray.v = rtc_ray.v;
  ray.dist = rtc_ray.tfar;
  ray.triangle_id = rtc_ray.primID;
  return false;
}

TC_IMPLEMENTATION(RayIntersection, BruteForceRayIntersection, "bf");

TC_IMPLEMENTATION(RayIntersection, EmbreeRayIntersection, "embree");

TC_NAMESPACE_END
