#include <cstdio>
using namespace std;
extern "C" {
void add_and_mul(float a, float b, float *c, float *d, int *e) {
    *c = a + b;
    *d = a * b;
    *e = int(a * b + a);
}
}
