extern "C" {
float add(float a, float b) {
	return a + b;
}

void add_and_mul(float a, float b, float *c, float *d) {
    *c = a + b;
    *d = a * b;
}
}
