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

#ifndef OPENVDB_VIEWER_CLIPBOX_HAS_BEEN_INCLUDED
#define OPENVDB_VIEWER_CLIPBOX_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif


namespace openvdb_viewer {

class ClipBox
{
public:
    ClipBox();

    void enableClipping() const;
    void disableClipping() const;

    void setBBox(const openvdb::BBoxd&);
    void setStepSize(const openvdb::Vec3d& s) { mStepSize = s; }

    void render();

    void update(double steps);
    void reset();

    bool isActive() const { return (mXIsActive || mYIsActive ||mZIsActive); }

    bool& activateXPlanes() { return mXIsActive;  }
    bool& activateYPlanes() { return mYIsActive;  }
    bool& activateZPlanes() { return mZIsActive;  }

    bool& shiftIsDown() { return mShiftIsDown; }
    bool& ctrlIsDown() { return mCtrlIsDown; }

    bool mouseButtonCallback(int button, int action);
    bool mousePosCallback(int x, int y);

private:
    void update() const;

    openvdb::Vec3d mStepSize;
    openvdb::BBoxd mBBox;
    bool mXIsActive, mYIsActive, mZIsActive, mShiftIsDown, mCtrlIsDown;
    GLdouble mFrontPlane[4], mBackPlane[4], mLeftPlane[4], mRightPlane[4],
        mTopPlane[4], mBottomPlane[4];
}; // class ClipBox

} // namespace openvdb_viewer

#endif // OPENVDB_VIEWER_CLIPBOX_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
