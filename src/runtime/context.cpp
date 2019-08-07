struct Context {
  int a, b, c;
};

extern "C" int test(Context context) {
  return 1;
}
