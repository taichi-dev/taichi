#include <cstdio>
#include <cstdlib>
#include <string>
#include <Python.h>
#include "taichi/platform/windows/windows.h"
#include <iostream>
#include <vector>
#include <string>

void main(int argc, char **argv) {
  Py_SetProgramName(L"ti");
  Py_Initialize();

  std::vector<std::wstring> argv_converted;
  std::vector<wchar_t *> argv_char;
  argv_converted.resize(argc);
  argv_char.resize(argc);

  for (int i = 0; i < argc; i++) {
    int buffer_len = 3 * std::strlen(argv[i]) + 2;
    // printf("len %d\n", buffer_len);
    // would rather be safe here... TODO: figure out the maximum converted
    // length
    argv_converted[i].resize(buffer_len);
    auto ret = mbstowcs(&argv_converted[i][0], argv[i], buffer_len);
    argv_char[i] = &argv_converted[i][0];
  }
  PySys_SetArgv(argc, &argv_char[0]);
  // TODO: implement release mode for this
  auto dir = getenv("TAICHI_REPO_DIR");
  if (dir == nullptr) {
    std::cout << "Please set TAICHI_REPO_DIR" << std::endl;
    exit(-1);
  }
  auto path = std::string(dir) + "/bin/ti";
  auto file = std::fopen(path.c_str(), "r");
  PyRun_SimpleFile(file, "ti");
  Py_Finalize();
  return;
}