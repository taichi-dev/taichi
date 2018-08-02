#include <mkl.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <unistd.h>
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
  for (int i = 0; ; i++) {
    if (i > 5) {
    }
    double t = get_time();
    cblas_saxpy(n, 2, x.data(), 1, y.data(), 1);
    printf("%f\n", get_time() - t);
  }
  return 0;
}