#include <mkl.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <cstring>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}
constexpr int n = 1024 * 1024 * 1024;

int main() {
  auto x = std::vector<float>(n), y = std::vector<float>(n);
  for (int i = 0; i < n; i++) {
    x[i] = i;
    y[i] = n - i * 2;
  }
  cblas_saxpy(n, 2, x.data(), 1, y.data(), 1);
  for (int i = 0; i < n; i++) {
    if (std::abs(y[i] - n) > n * 1e-6f) {
      printf("Error!\n");
      exit(-1);
    }
  }
  printf("Correct!\n");
  float trash = 0;
  for (int i = 0; i < 10000000; i++) {
    if (i > 5) {
    }
    double t = get_time();
    cblas_saxpy(n, 2, x.data(), 1, y.data(), 1);
    auto saxpy = get_time() - t;
    t = get_time();
    trash += cblas_sasum(n, x.data(), 1);
    double sasum = get_time() - t;
    t = get_time();
    cblas_sscal(n, 0.99f, x.data(), 1);
    double sscal = get_time() - t;
    t = get_time();
    std::memset(x.data(), 0, sizeof(float) * n);
    double memset = get_time() - t;
    printf("saxpy %f sasum %f sscal %f memset %f\n", saxpy, sasum, sscal,
           memset);
  }
  printf("trash %f", trash);
  return 0;
}