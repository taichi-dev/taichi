#include <vector>
#include <Setting.h>
#include <stdlib.h>
#include <Utility/PoissonDisk/SampleGenerator.h>
namespace mn {
template <typename T, int Scheme>
auto Initialize_Data();
template <>
auto Initialize_Data<T, 0>() {  ///< cube
  std::vector<std::array<T, Dim>> data;
  static const int min = N / 8;
  static const int max = N / 8 * 7;
  static const T dx = (T)1. / (T)N;
  for (int i = min; i < max; ++i)
    for (int j = min; j < max; ++j)
      for (int k = min; k < max; ++k) {
        std::array<T, Dim> cell_center;
        cell_center[0] = (i + 0.5) * dx;
        cell_center[1] = (j + 0.5) * dx;
        cell_center[2] = (k + 0.5) * dx;
        for (int ii = -1; ii <= 1; ii = ii + 2)
          for (int jj = -1; jj <= 1; jj = jj + 2)
            for (int kk = -1; kk <= 1; kk = kk + 2) {
              std::array<T, Dim> particle;
              particle[0] = cell_center[0] + ii * 0.25 * dx;
              particle[1] = cell_center[1] + jj * 0.25 * dx;
              particle[2] = cell_center[2] + kk * 0.25 * dx;
              data.push_back(particle);
            }
      }
  return data;
}
template <>
auto Initialize_Data<T, 7>() {
  std::vector<std::array<T, Dim>> data;
  int center[3] = {N / 2, N / 2, N / 2};
  int res[3] = {5 * N / 6, 5 * N / 6, 5 * N / 6};
  int minCorner[3];
  for (int i = 0; i < 3; ++i)
    minCorner[i] = center[i] - .5 * res[i];
  minCorner[1] = center[1] + 0.25 * res[1];
  std::string fileName = "../Assets/two_dragons.sdf";
  int samplePerCell = 20;
  int offsetx = minCorner[0];
  int offsety = minCorner[1];
  int offsetz = minCorner[2];
  int width = res[0];
  int height = res[1];
  int depth = res[2];
  float levelsetDx;
  SampleGenerator pd;
  std::vector<float> samples;
  float levesetMinx, levelsetMiny, levelsetMinz;
  int levelsetNi, levelsetNj, levelsetNk;
  pd.LoadSDF(fileName, levelsetDx, levesetMinx, levelsetMiny, levelsetMinz,
             levelsetNi, levelsetNj, levelsetNk);
  int minx = 1, miny = 1, minz = 1;
  int maxx = levelsetNi - 2, maxy = levelsetNj - 2, maxz = levelsetNk - 2;
  float scalex = 1.f * width / (maxx - minx);
  float scaley = 1.f * height / (maxy - miny);
  float scalez = 1.f * depth / (maxz - minz);
  float scale = scalex < scaley ? scalex : scaley;
  scale = scalez < scale ? scalez : scale;
  float samplePerLevelsetCell = samplePerCell * scale * scale * scale;
  pd.GenerateUniformSamples(samplePerLevelsetCell, samples);
  for (int i = 0, size = samples.size() / 3; i < size; i++) {
    std::array<T, Dim> particle;
    particle[0] = ((samples[i * 3 + 0] - minx) * scale + offsetx + 4) * dx;
    particle[1] = ((samples[i * 3 + 1] - miny) * scale + offsety + 4) * dx;
    particle[2] = ((samples[i * 3 + 2] - minz) * scale + offsetz + 4) * dx;
    data.push_back(particle);
  }
  return data;
}
}  // namespace mn
