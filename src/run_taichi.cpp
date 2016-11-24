#include <iostream>
#ifdef WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "math/array_3d.h"
#include "system/timer.h"

int main() {
    using std::cout;
    using std::cerr;

	typedef int(*TaichiMain)(void);

    // load the triangle library
#ifdef WIN32
	HINSTANCE taichi = LoadLibrary("./taichi_dll.dll");
	if (!taichi) {
		cerr << "Cannot load library: " << '\n';
		return 1;
	}
#else
	void *taichi = dlopen("./output/libtaichi_dll.dylib", RTLD_LAZY);
	if (!taichi) {
		cerr << "Cannot load library: " << dlerror() << '\n';
		return 1;
	}
#endif
#ifdef WIN32
    auto taichi_main = (TaichiMain) GetProcAddress(taichi, "taichi_main");
	assert_info(taichi_main != nullptr, "taichi_main not found");
#else
    // load the symbols
    auto taichi_main = (TaichiMain) dlsym(taichi, "taichi_main");
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        cerr << "Cannot load symbol main: " << dlsym_error << '\n';
        return 1;
    }
#endif
    taichi_main();
#ifdef WIN32
	
#else
    dlclose(taichi);
#endif
	return 0;
}

