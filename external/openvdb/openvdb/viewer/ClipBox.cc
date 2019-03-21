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

#include "ClipBox.h"


namespace openvdb_viewer {

ClipBox::ClipBox()
    : mStepSize(1.0)
    , mBBox()
    , mXIsActive(false)
    , mYIsActive(false)
    , mZIsActive(false)
    , mShiftIsDown(false)
    , mCtrlIsDown(false)
{
    GLdouble front [] = { 0.0, 0.0, 1.0, 0.0};
    std::copy(front, front + 4, mFrontPlane);

    GLdouble back [] = { 0.0, 0.0,-1.0, 0.0};
    std::copy(back, back + 4, mBackPlane);

    GLdouble left [] = { 1.0, 0.0, 0.0, 0.0};
    std::copy(left, left + 4, mLeftPlane);

    GLdouble right [] = {-1.0, 0.0, 0.0, 0.0};
    std::copy(right, right + 4, mRightPlane);

    GLdouble top [] = { 0.0, 1.0, 0.0, 0.0};
    std::copy(top, top + 4, mTopPlane);

    GLdouble bottom [] = { 0.0,-1.0, 0.0, 0.0};
    std::copy(bottom, bottom + 4, mBottomPlane);
}


void
ClipBox::setBBox(const openvdb::BBoxd& bbox)
{
    mBBox = bbox;
    reset();
}


void
ClipBox::update(double steps)
{
    if (mXIsActive) {
        GLdouble s = steps * mStepSize.x() * 4.0;

        if (mShiftIsDown || mCtrlIsDown) {
            mLeftPlane[3] -= s;
            mLeftPlane[3] = -std::min(-mLeftPlane[3], (mRightPlane[3] - mStepSize.x()));
            mLeftPlane[3] = -std::max(-mLeftPlane[3], mBBox.min().x());
        }

        if (!mShiftIsDown || mCtrlIsDown) {
            mRightPlane[3] += s;
            mRightPlane[3] = std::min(mRightPlane[3], mBBox.max().x());
            mRightPlane[3] = std::max(mRightPlane[3], (-mLeftPlane[3] + mStepSize.x()));
        }
    }

    if (mYIsActive) {
        GLdouble s = steps * mStepSize.y() * 4.0;

        if (mShiftIsDown || mCtrlIsDown) {
            mTopPlane[3] -= s;
            mTopPlane[3] = -std::min(-mTopPlane[3], (mBottomPlane[3] - mStepSize.y()));
            mTopPlane[3] = -std::max(-mTopPlane[3], mBBox.min().y());
        }

        if (!mShiftIsDown || mCtrlIsDown) {
            mBottomPlane[3] += s;
            mBottomPlane[3] = std::min(mBottomPlane[3], mBBox.max().y());
            mBottomPlane[3] = std::max(mBottomPlane[3], (-mTopPlane[3] + mStepSize.y()));
        }
    }

    if (mZIsActive) {
        GLdouble s = steps * mStepSize.z() * 4.0;

        if (mShiftIsDown || mCtrlIsDown) {
            mFrontPlane[3] -= s;
            mFrontPlane[3] = -std::min(-mFrontPlane[3], (mBackPlane[3] - mStepSize.z()));
            mFrontPlane[3] = -std::max(-mFrontPlane[3], mBBox.min().z());
        }

        if (!mShiftIsDown || mCtrlIsDown) {
            mBackPlane[3] += s;
            mBackPlane[3] = std::min(mBackPlane[3], mBBox.max().z());
            mBackPlane[3] = std::max(mBackPlane[3], (-mFrontPlane[3] + mStepSize.z()));
        }
    }
}


void
ClipBox::reset()
{
    mFrontPlane[3] = std::abs(mBBox.min().z());
    mBackPlane[3] = mBBox.max().z();

    mLeftPlane[3] = std::abs(mBBox.min().x());
    mRightPlane[3] = mBBox.max().x();

    mTopPlane[3] = std::abs(mBBox.min().y());
    mBottomPlane[3] = mBBox.max().y();
}


void
ClipBox::update() const
{
    glClipPlane(GL_CLIP_PLANE0, mFrontPlane);
    glClipPlane(GL_CLIP_PLANE1, mBackPlane);
    glClipPlane(GL_CLIP_PLANE2, mLeftPlane);
    glClipPlane(GL_CLIP_PLANE3, mRightPlane);
    glClipPlane(GL_CLIP_PLANE4, mTopPlane);
    glClipPlane(GL_CLIP_PLANE5, mBottomPlane);
}


void
ClipBox::enableClipping() const
{
    update();
    if (-mFrontPlane[3] > mBBox.min().z())  glEnable(GL_CLIP_PLANE0);
    if (mBackPlane[3] < mBBox.max().z())    glEnable(GL_CLIP_PLANE1);
    if (-mLeftPlane[3] > mBBox.min().x())   glEnable(GL_CLIP_PLANE2);
    if (mRightPlane[3] < mBBox.max().x())   glEnable(GL_CLIP_PLANE3);
    if (-mTopPlane[3] > mBBox.min().y())    glEnable(GL_CLIP_PLANE4);
    if (mBottomPlane[3] < mBBox.max().y())  glEnable(GL_CLIP_PLANE5);
}


void
ClipBox::disableClipping() const
{
    glDisable(GL_CLIP_PLANE0);
    glDisable(GL_CLIP_PLANE1);
    glDisable(GL_CLIP_PLANE2);
    glDisable(GL_CLIP_PLANE3);
    glDisable(GL_CLIP_PLANE4);
    glDisable(GL_CLIP_PLANE5);
}


void
ClipBox::render()
{
    bool drawBbox = false;

    const GLenum geoMode = GL_LINE_LOOP;

    glColor3d(0.1, 0.1, 0.9);
    if (-mFrontPlane[3] > mBBox.min().z()) {
        glBegin(geoMode);
        glVertex3d(mBBox.min().x(), mBBox.min().y(), -mFrontPlane[3]);
        glVertex3d(mBBox.min().x(), mBBox.max().y(), -mFrontPlane[3]);
        glVertex3d(mBBox.max().x(), mBBox.max().y(), -mFrontPlane[3]);
        glVertex3d(mBBox.max().x(), mBBox.min().y(), -mFrontPlane[3]);
        glEnd();
        drawBbox = true;
    }

    if (mBackPlane[3] < mBBox.max().z()) {
        glBegin(geoMode);
        glVertex3d(mBBox.min().x(), mBBox.min().y(), mBackPlane[3]);
        glVertex3d(mBBox.min().x(), mBBox.max().y(), mBackPlane[3]);
        glVertex3d(mBBox.max().x(), mBBox.max().y(), mBackPlane[3]);
        glVertex3d(mBBox.max().x(), mBBox.min().y(), mBackPlane[3]);
        glEnd();
        drawBbox = true;
    }

    glColor3d(0.9, 0.1, 0.1);
    if (-mLeftPlane[3] > mBBox.min().x()) {
        glBegin(geoMode);
        glVertex3d(-mLeftPlane[3], mBBox.min().y(), mBBox.min().z());
        glVertex3d(-mLeftPlane[3], mBBox.max().y(), mBBox.min().z());
        glVertex3d(-mLeftPlane[3], mBBox.max().y(), mBBox.max().z());
        glVertex3d(-mLeftPlane[3], mBBox.min().y(), mBBox.max().z());
        glEnd();
        drawBbox = true;
    }

    if (mRightPlane[3] < mBBox.max().x()) {
        glBegin(geoMode);
        glVertex3d(mRightPlane[3], mBBox.min().y(), mBBox.min().z());
        glVertex3d(mRightPlane[3], mBBox.max().y(), mBBox.min().z());
        glVertex3d(mRightPlane[3], mBBox.max().y(), mBBox.max().z());
        glVertex3d(mRightPlane[3], mBBox.min().y(), mBBox.max().z());
        glEnd();
        drawBbox = true;
    }

    glColor3d(0.1, 0.9, 0.1);
    if (-mTopPlane[3] > mBBox.min().y()) {
        glBegin(geoMode);
        glVertex3d(mBBox.min().x(), -mTopPlane[3], mBBox.min().z());
        glVertex3d(mBBox.min().x(), -mTopPlane[3], mBBox.max().z());
        glVertex3d(mBBox.max().x(), -mTopPlane[3], mBBox.max().z());
        glVertex3d(mBBox.max().x(), -mTopPlane[3], mBBox.min().z());
        glEnd();
        drawBbox = true;
    }

    if (mBottomPlane[3] < mBBox.max().y()) {
        glBegin(geoMode);
        glVertex3d(mBBox.min().x(), mBottomPlane[3], mBBox.min().z());
        glVertex3d(mBBox.min().x(), mBottomPlane[3], mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBottomPlane[3], mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBottomPlane[3], mBBox.min().z());
        glEnd();
        drawBbox = true;
    }

    if (drawBbox) {
        glColor3d(0.5, 0.5, 0.5);
        glBegin(GL_LINE_LOOP);
        glVertex3d(mBBox.min().x(), mBBox.min().y(), mBBox.min().z());
        glVertex3d(mBBox.min().x(), mBBox.min().y(), mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBBox.min().y(), mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBBox.min().y(), mBBox.min().z());
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3d(mBBox.min().x(), mBBox.max().y(), mBBox.min().z());
        glVertex3d(mBBox.min().x(), mBBox.max().y(), mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBBox.max().y(), mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBBox.max().y(), mBBox.min().z());
        glEnd();

        glBegin(GL_LINES);
        glVertex3d(mBBox.min().x(), mBBox.min().y(), mBBox.min().z());
        glVertex3d(mBBox.min().x(), mBBox.max().y(), mBBox.min().z());
        glVertex3d(mBBox.min().x(), mBBox.min().y(), mBBox.max().z());
        glVertex3d(mBBox.min().x(), mBBox.max().y(), mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBBox.min().y(), mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBBox.max().y(), mBBox.max().z());
        glVertex3d(mBBox.max().x(), mBBox.min().y(), mBBox.min().z());
        glVertex3d(mBBox.max().x(), mBBox.max().y(), mBBox.min().z());
        glEnd();
    }
}


////////////////////////////////////////


bool
ClipBox::mouseButtonCallback(int /*button*/, int /*action*/)
{
    return false; // unhandled
}


bool
ClipBox::mousePosCallback(int /*x*/, int /*y*/)
{
    return false; // unhandled
}

} // namespace openvdb_viewer

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
