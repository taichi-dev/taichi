///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include "Camera.h"

#include <cmath>

#ifdef OPENVDB_USE_GLFW_3
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#else // if !defined(OPENVDB_USE_GLFW_3)
#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include <GL/glfw.h>
#endif // !defined(OPENVDB_USE_GLFW_3)


namespace openvdb_viewer {

const double Camera::sDeg2rad = M_PI / 180.0;


Camera::Camera()
    : mFov(65.0)
    , mNearPlane(0.1)
    , mFarPlane(10000.0)
    , mTarget(openvdb::Vec3d(0.0))
    , mLookAt(mTarget)
    , mUp(openvdb::Vec3d(0.0, 1.0, 0.0))
    , mForward(openvdb::Vec3d(0.0, 0.0, 1.0))
    , mRight(openvdb::Vec3d(1.0, 0.0, 0.0))
    , mEye(openvdb::Vec3d(0.0, 0.0, -1.0))
    , mTumblingSpeed(0.5)
    , mZoomSpeed(0.2)
    , mStrafeSpeed(0.05)
    , mHead(30.0)
    , mPitch(45.0)
    , mTargetDistance(25.0)
    , mDistance(mTargetDistance)
    , mMouseDown(false)
    , mStartTumbling(false)
    , mZoomMode(false)
    , mChanged(true)
    , mNeedsDisplay(true)
    , mMouseXPos(0.0)
    , mMouseYPos(0.0)
#if GLFW_VERSION_MAJOR >= 3
    , mWindow(nullptr)
#endif
{
}


void
Camera::lookAt(const openvdb::Vec3d& p, double dist)
{
    mLookAt = p;
    mDistance = dist;
    mNeedsDisplay = true;
}


void
Camera::lookAtTarget()
{
    mLookAt = mTarget;
    mDistance = mTargetDistance;
    mNeedsDisplay = true;
}


void
Camera::setSpeed(double zoomSpeed, double strafeSpeed, double tumblingSpeed)
{
    mZoomSpeed = std::max(0.0001, mDistance * zoomSpeed);
    mStrafeSpeed = std::max(0.0001, mDistance * strafeSpeed);
    mTumblingSpeed = std::max(0.2, mDistance * tumblingSpeed);
    mTumblingSpeed = std::min(1.0, mDistance * tumblingSpeed);
}


void
Camera::setTarget(const openvdb::Vec3d& p, double dist)
{
    mTarget = p;
    mTargetDistance = dist;
}


void
Camera::aim()
{
#if GLFW_VERSION_MAJOR >= 3
    if (mWindow == nullptr) return;
#endif

    // Get the window size
    int width, height;
#if GLFW_VERSION_MAJOR >= 3
    glfwGetFramebufferSize(mWindow, &width, &height);
#else
    glfwGetWindowSize(&width, &height);
#endif

    // Make sure that height is non-zero to avoid division by zero
    height = std::max(1, height);

    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Window aspect (assumes square pixels)
    double aspectRatio = double(width) / double(height);

    // Set perspective view (fov is in degrees in the y direction.)
    gluPerspective(mFov, aspectRatio, mNearPlane, mFarPlane);

    if (mChanged) {

        mChanged = false;

        mEye[0] = mLookAt[0] + mDistance * std::cos(mHead * sDeg2rad) * std::cos(mPitch * sDeg2rad);
        mEye[1] = mLookAt[1] + mDistance * std::sin(mHead * sDeg2rad);
        mEye[2] = mLookAt[2] + mDistance * std::cos(mHead * sDeg2rad) * std::sin(mPitch * sDeg2rad);

        mForward = mLookAt - mEye;
        mForward.normalize();

        mUp[1] = std::cos(mHead * sDeg2rad) > 0 ? 1.0 : -1.0;
        mRight = mForward.cross(mUp);
    }

    // Set up modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(mEye[0], mEye[1], mEye[2],
              mLookAt[0], mLookAt[1], mLookAt[2],
              mUp[0], mUp[1], mUp[2]);
    mNeedsDisplay = false;
}


void
Camera::keyCallback(int key, int)
{
#if GLFW_VERSION_MAJOR >= 3
    if (mWindow == nullptr) return;
    int state = glfwGetKey(mWindow, key);
#else
    int state = glfwGetKey(key);
#endif
    switch (state) {
        case GLFW_PRESS:
            switch(key) {
                case GLFW_KEY_SPACE:
                    mZoomMode = true;
                    break;
            }
            break;
        case GLFW_RELEASE:
            switch(key) {
                case GLFW_KEY_SPACE:
                    mZoomMode = false;
                    break;
            }
            break;
    }
    mChanged = true;
}


void
Camera::mouseButtonCallback(int button, int action)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) mMouseDown = true;
        else if (action == GLFW_RELEASE) mMouseDown = false;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mMouseDown = true;
            mZoomMode = true;
        } else if (action == GLFW_RELEASE) {
            mMouseDown = false;
            mZoomMode = false;
        }
    }
    if (action == GLFW_RELEASE) mMouseDown = false;

    mStartTumbling = true;
    mChanged = true;
}


void
Camera::mousePosCallback(int x, int y)
{
    if (mStartTumbling) {
        mMouseXPos = x;
        mMouseYPos = y;
        mStartTumbling = false;
    }

    double dx, dy;
    dx = x - mMouseXPos;
    dy = y - mMouseYPos;

    if (mMouseDown && !mZoomMode) {
        mNeedsDisplay = true;
        mHead += dy * mTumblingSpeed;
        mPitch += dx * mTumblingSpeed;
    } else if (mMouseDown && mZoomMode) {
        mNeedsDisplay = true;
        mLookAt += (dy * mUp - dx * mRight) * mStrafeSpeed;
    }

    mMouseXPos = x;
    mMouseYPos = y;
    mChanged = true;
}


void
Camera::mouseWheelCallback(int pos, int prevPos)
{
    double speed = std::abs(prevPos - pos);

    if (prevPos < pos) {
        mDistance += speed * mZoomSpeed;
    } else {
        double temp = mDistance - speed * mZoomSpeed;
        mDistance = std::max(0.0, temp);
    }
    setSpeed();

    mChanged = true;
    mNeedsDisplay = true;
}

} // namespace openvdb_viewer

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
