#pragma once

#include "config.h"

namespace boost {
    namespace python {
        class dict;
    }
}

TC_NAMESPACE_BEGIN

    class Simulator {
    public:
        virtual void initialize(const Config &config) {};

        virtual void step(float delta_t) {};

        //virtual void add_particle(const Config &config) {
        //	error("Not implemented");
        //}
        virtual boost::python::dict get_simulation_data(const Config &config);
    };

TC_NAMESPACE_END
