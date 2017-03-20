/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/camera.h>
#include <taichi/dynamics/simulation3d.h>
#include <taichi/visual/texture.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(Camera, "Camera");
TC_INTERFACE_DEF(Simulation3D, "simulation3d")
TC_INTERFACE_DEF(Texture, "texture");

TC_NAMESPACE_END
