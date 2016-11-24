#include "taichi.h"

TC_NAMESPACE_BEGIN

void run_demo(Config config) {
	string target = config.get_string("target");
	static std::map<std::string, std::function<void(Config)>> targets = {
		{"rendering", rendering_demo},
		{"heat", heat_demo},
		{"spectrum", spectrum_demo},
		{"hdr", hdr_demo},
		{"fluid", fluid_demo},
		{"point_cloud", point_cloud_demo}
	};
	if (targets.find(target) != targets.end()) {
		targets[target](Config::load(target + "_demo.cfg"));
	}
	else {
		error(string("Target ") + target + " not found!");
	}
}


TC_NAMESPACE_END

using namespace taichi;

#ifdef WIN32
#define TC_EXPORT extern "C" __declspec(dllexport) 
#else
#define TC_EXPORT extern "C"
#endif

TC_EXPORT int taichi_main() {
	try {
		run_demo(Config::load("taichi.cfg"));
	}
	catch (const std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
		return 1;
	}
	catch (...) {
		std::cout << "Error: unknown exception caught." << std::endl;
		return 1;
	}

	return 0;
}


