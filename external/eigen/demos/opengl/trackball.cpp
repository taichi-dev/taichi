// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "trackball.h"
#include "camera.h"

using namespace Eigen;

void Trackball::track(const Vector2i& point2D)
{
  if (mpCamera==0)
    return;
  Vector3f newPoint3D;
  bool newPointOk = mapToSphere(point2D, newPoint3D);

  if (mLastPointOk && newPointOk)
  {
    Vector3f axis = mLastPoint3D.cross(newPoint3D).normalized();
    float cos_angle = mLastPoint3D.dot(newPoint3D);
    if ( std::abs(cos_angle) < 1.0 )
    {
      float angle = 2. * acos(cos_angle);
      if (mMode==Around)
        mpCamera->rotateAroundTarget(Quaternionf(AngleAxisf(angle, axis)));
      else
        mpCamera->localRotate(Quaternionf(AngleAxisf(-angle, axis)));
    }
  }

  mLastPoint3D = newPoint3D;
  mLastPointOk = newPointOk;
}

bool Trackball::mapToSphere(const Vector2i& p2, Vector3f& v3)
{
  if ((p2.x() >= 0) && (p2.x() <= int(mpCamera->vpWidth())) &&
      (p2.y() >= 0) && (p2.y() <= int(mpCamera->vpHeight())) )
  {
    double x  = (double)(p2.x() - 0.5*mpCamera->vpWidth())  / (double)mpCamera->vpWidth();
    double y  = (double)(0.5*mpCamera->vpHeight() - p2.y()) / (double)mpCamera->vpHeight();
    double sinx         = sin(M_PI * x * 0.5);
    double siny         = sin(M_PI * y * 0.5);
    double sinx2siny2   = sinx * sinx + siny * siny;

    v3.x() = sinx;
    v3.y() = siny;
    v3.z() = sinx2siny2 < 1.0 ? sqrt(1.0 - sinx2siny2) : 0.0;

    return true;
  }
  else
    return false;
}
