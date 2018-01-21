/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "framebuffer.h"

TC_NAMESPACE_BEGIN

class NaiveFramebuffer : public Framebuffer {};

TC_IMPLEMENTATION(Framebuffer, NaiveFramebuffer, "naive");

TC_NAMESPACE_END
