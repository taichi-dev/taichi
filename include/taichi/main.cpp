/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

///////////////////////////
///       _.----._
///      /      \#\
///     |  T    /##\
///     |      ####|
///     \    /###C#/
///      \_  \####/
///        ''--''
///////////////////////////



#include <taichi/common/util.h>
#include <taichi/common/interface.h>
#include <taichi/common/task.h>
#include <iostream>

TC_NAMESPACE_BEGIN

int test_linalg();

TC_NAMESPACE_END

int main(int argc, char **argv) {
    using namespace taichi;
    std::cout
            << "                          *******                          " << std::endl
            << " ********************************************************* " << std::endl
            << " ** Taichi - Physically based Computer Graphics Library ** " << std::endl
            << " ********************************************************* " << std::endl
            << "                          *******                          " << std::endl
            << std::endl;
    if (argc == 1) {
        std::cout << "    usage: taichi run [task name]" << std::endl;
        std::cout << "           taichi test [module name]" << std::endl;
        return -1;
    }
    std::string mode = argv[1];
    if (mode == "run") {
        if (argc <= 2) {
            std::cout << "Please specify [task name], e.g. test_math" << std::endl;
            return -1;
        }
        std::vector<std::string> parameters;
        std::string task_name(argv[2]);
        for (int i = 3; i < argc; i++) {
            parameters.push_back(std::string(argv[i]));
        }
        auto task = create_instance<Task>(task_name);
        task->run(parameters);
    } else if (mode == "test") {
        NOT_IMPLEMENTED
    } else {
        error("Mode should be 'run' or 'test' instead of " + mode);
    }
    return 0;
}

