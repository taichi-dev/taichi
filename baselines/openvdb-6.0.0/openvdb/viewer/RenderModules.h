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

#ifndef OPENVDB_VIEWER_RENDERMODULES_HAS_BEEN_INCLUDED
#define OPENVDB_VIEWER_RENDERMODULES_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/math/Operators.h>
#include <string>
#include <vector>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif


namespace openvdb_viewer {

// OpenGL helper objects

class BufferObject
{
public:
    BufferObject();
    ~BufferObject();

    void render() const;

    /// @note accepted @c primType: GL_POINTS, GL_LINE_STRIP, GL_LINE_LOOP,
    /// GL_LINES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_TRIANGLES,
    /// GL_QUAD_STRIP, GL_QUADS and GL_POLYGON
    void genIndexBuffer(const std::vector<GLuint>&, GLenum primType);

    void genVertexBuffer(const std::vector<GLfloat>&);
    void genNormalBuffer(const std::vector<GLfloat>&);
    void genColorBuffer(const std::vector<GLfloat>&);

    void clear();

private:
    GLuint mVertexBuffer, mNormalBuffer, mIndexBuffer, mColorBuffer;
    GLenum mPrimType;
    GLsizei mPrimNum;
};


class ShaderProgram
{
public:
    ShaderProgram();
    ~ShaderProgram();

    void setVertShader(const std::string&);
    void setFragShader(const std::string&);

    void build();
    void build(const std::vector<GLchar*>& attributes);

    void startShading() const;
    void stopShading() const;

    void clear();

private:
    GLuint mProgram, mVertShader, mFragShader;
};


////////////////////////////////////////


/// @brief interface class
class RenderModule
{
public:
    virtual ~RenderModule() {}

    virtual void render() = 0;

    bool visible() { return mIsVisible; }
    void setVisible(bool b) { mIsVisible = b; }

protected:
    RenderModule(): mIsVisible(true) {}

    bool mIsVisible;
};


////////////////////////////////////////


/// @brief Basic render module, axis gnomon and ground plane.
class ViewportModule: public RenderModule
{
public:
    ViewportModule();
    ~ViewportModule() override = default;

    void render() override;

private:
    float mAxisGnomonScale, mGroundPlaneScale;
};


////////////////////////////////////////


/// @brief Tree topology render module
class TreeTopologyModule: public RenderModule
{
public:
    TreeTopologyModule(const openvdb::GridBase::ConstPtr&);
    ~TreeTopologyModule() override = default;

    void render() override;

private:
    void init();

    const openvdb::GridBase::ConstPtr& mGrid;
    BufferObject mBufferObject;
    bool mIsInitialized;
    ShaderProgram mShader;
};


////////////////////////////////////////


/// @brief Module to render active voxels as points
class VoxelModule: public RenderModule
{
public:
    VoxelModule(const openvdb::GridBase::ConstPtr&);
    ~VoxelModule() override = default;

    void render() override;

private:
    void init();

    const openvdb::GridBase::ConstPtr& mGrid;
    BufferObject mInteriorBuffer, mSurfaceBuffer, mVectorBuffer;
    bool mIsInitialized;
    ShaderProgram mFlatShader, mSurfaceShader;
};


////////////////////////////////////////


/// @brief Surfacing render module
class MeshModule: public RenderModule
{
public:
    MeshModule(const openvdb::GridBase::ConstPtr&);
    ~MeshModule() override = default;

    void render() override;

private:
    void init();

    const openvdb::GridBase::ConstPtr& mGrid;
    BufferObject mBufferObject;
    bool mIsInitialized;
    ShaderProgram mShader;
};

} // namespace openvdb_viewer

#endif // OPENVDB_VIEWER_RENDERMODULES_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
