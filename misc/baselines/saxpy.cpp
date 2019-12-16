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
    t = get_time();
    std::memcpy(x.data(), y.data(), sizeof(float) * n);
    double memcpy = get_time() - t;
    printf(
        "time saxpy(12) %f sasum(4) %f sscal(8) %f memset(4) %f memcpy(8) %f\n",
        saxpy, sasum, sscal, memset, memcpy);
    printf(
        "BW:  saxpy(12) %.2f sasum(4) %.2f sscal(8) %.2f memset(4) %.2f "
        "memcpy(8) %.2f\n",
        12.0f / saxpy, 4.0f / sasum, 8.0f / sscal, 4.0f / memset,
        8.0f / memcpy);
  }
  printf("trash %f", trash);
  return 0;
}
