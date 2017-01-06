#include <taichi/python/exception.h>

TC_NAMESPACE_BEGIN

void raise_assertion_failure_in_python(const std::string &msg) {
    throw ExceptionForPython(msg);
}

TC_NAMESPACE_END

void taichi_raise_assertion_failure_in_python(const char *msg) {
    return taichi::raise_assertion_failure_in_python(std::string(msg));
}

