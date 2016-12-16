#include "fluid.h"

#include "euler_fluid.h"
#include "euler_smoke.h"
#include "flip_fluid.h"
#include "voronoi_flip_fluid.h"
#include "flip_smoke.h"
#include "apic.h"

TC_NAMESPACE_BEGIN

long long Fluid::Particle::instance_counter = 0;

std::shared_ptr<Fluid> create_fluid(std::string name)
{
	std::shared_ptr<Fluid> fluid;
	if (name == "flip")
		fluid = std::make_shared<FLIPFluid>();
	else if (name == "euler")
		fluid = std::make_shared<EulerFluid>();
	else if (name == "apic")
		fluid = std::make_shared<APICFluid>();
	else if (name == "vor")
		fluid = std::make_shared<VoronoiFLIP>();
	else if (name == "flip_smoke")
		fluid = std::make_shared<FLIPSmoke>();
	else if (name == "flip_vor_smoke")
		fluid = std::make_shared<VoronoiFLIPSmoke>();
	else
		error("Unknown Simulator Name");
	return fluid;
}

TC_NAMESPACE_END

