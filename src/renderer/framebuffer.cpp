#include "camera.h"

TC_NAMESPACE_BEGIN
    TC_INTERFACE_DEF(Framebuffer, "framebuffer");

    class NaiveFramebuffer : public Framebuffer {

    };

    TC_IMPLEMENTATION(Framebuffer, NaiveFramebuffer, "naive");
TC_NAMESPACE_END

