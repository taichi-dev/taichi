#include <cstdio>
#include <cstdlib>
#include <string>
#include <Python.h>
#include <windows.h>
#include <iostream>

void main(int argc, wchar_t *argv[]) {
    Py_SetProgramName(L"ti");
    Py_Initialize();
    PySys_SetArgv(argc, argv);
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