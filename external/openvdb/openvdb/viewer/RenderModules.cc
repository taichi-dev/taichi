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

#include "RenderModules.h"

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/util/logging.h>
#include <algorithm> // for std::min()
#include <cmath> // for std::abs(), std::fabs(), std::floor()
#include <limits>
#include <type_traits> // for std::is_const


namespace openvdb_viewer {

namespace util {

/// Helper class used internally by processTypedGrid()
template<typename GridType, typename OpType, bool IsConst/*=false*/>
struct GridProcessor {
    static inline void call(OpType& op, openvdb::GridBase::Ptr grid) {
#ifdef _MSC_VER
        op.operator()<GridType>(openvdb::gridPtrCast<GridType>(grid));
#else
        op.template operator()<GridType>(openvdb::gridPtrCast<GridType>(grid));
#endif
    }
};

/// Helper class used internally by processTypedGrid()
template<typename GridType, typename OpType>
struct GridProcessor<GridType, OpType, /*IsConst=*/true> {
    static inline void call(OpType& op, openvdb::GridBase::ConstPtr grid) {
#ifdef _MSC_VER
        op.operator()<GridType>(openvdb::gridConstPtrCast<GridType>(grid));
#else
        op.template operator()<GridType>(openvdb::gridConstPtrCast<GridType>(grid));
#endif
    }
};


/// Helper function used internally by processTypedGrid()
template<typename GridType, typename OpType, typename GridPtrType>
inline void
doProcessTypedGrid(GridPtrType grid, OpType& op)
{
    GridProcessor<GridType, OpType,
        std::is_const<typename GridPtrType::element_type>::value>::call(op, grid);
}


////////////////////////////////////////


/// @brief Utility function that, given a generic grid pointer,
/// calls a functor on the fully-resolved grid
///
/// Usage:
/// @code
/// struct PruneOp {
///     template<typename GridT>
///     void operator()(typename GridT::Ptr grid) const { grid->tree()->prune(); }
/// };
///
/// processTypedGrid(myGridPtr, PruneOp());
/// @endcode
///
/// @return @c false if the grid type is unknown or unhandled.
template<typename GridPtrType, typename OpType>
bool
processTypedGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<BoolGrid>())        doProcessTypedGrid<BoolGrid>(grid, op);
    else if (grid->template isType<FloatGrid>())  doProcessTypedGrid<FloatGrid>(grid, op);
    else if (grid->template isType<DoubleGrid>()) doProcessTypedGrid<DoubleGrid>(grid, op);
    else if (grid->template isType<Int32Grid>())  doProcessTypedGrid<Int32Grid>(grid, op);
    else if (grid->template isType<Int64Grid>())  doProcessTypedGrid<Int64Grid>(grid, op);
    else if (grid->template isType<Vec3IGrid>())  doProcessTypedGrid<Vec3IGrid>(grid, op);
    else if (grid->template isType<Vec3SGrid>())  doProcessTypedGrid<Vec3SGrid>(grid, op);
    else if (grid->template isType<Vec3DGrid>())  doProcessTypedGrid<Vec3DGrid>(grid, op);
    else if (grid->template isType<points::PointDataGrid>()) {
        doProcessTypedGrid<points::PointDataGrid>(grid, op);
    }
    else return false;
    return true;
}


/// @brief Utility function that, given a generic grid pointer, calls
/// a functor on the fully-resolved grid, provided that the grid's
/// voxel values are scalars
template<typename GridPtrType, typename OpType>
bool
processTypedScalarGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<FloatGrid>())       doProcessTypedGrid<FloatGrid>(grid, op);
    else if (grid->template isType<DoubleGrid>()) doProcessTypedGrid<DoubleGrid>(grid, op);
    else if (grid->template isType<Int32Grid>())  doProcessTypedGrid<Int32Grid>(grid, op);
    else if (grid->template isType<Int64Grid>())  doProcessTypedGrid<Int64Grid>(grid, op);
    else return false;
    return true;
}


/// @brief Utility function that, given a generic grid pointer, calls
/// a functor on the fully-resolved grid, provided that the grid's
/// voxel values are scalars or PointIndex objects
template<typename GridPtrType, typename OpType>
bool
processTypedScalarOrPointDataGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (processTypedScalarGrid(grid, op)) return true;
    if (grid->template isType<points::PointDataGrid>()) {
        doProcessTypedGrid<points::PointDataGrid>(grid, op);
        return true;
    }
    return false;
}


/// @brief Utility function that, given a generic grid pointer, calls
/// a functor on the fully-resolved grid, provided that the grid's
/// voxel values are vectors
template<typename GridPtrType, typename OpType>
bool
processTypedVectorGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<Vec3IGrid>())       doProcessTypedGrid<Vec3IGrid>(grid, op);
    else if (grid->template isType<Vec3SGrid>())  doProcessTypedGrid<Vec3SGrid>(grid, op);
    else if (grid->template isType<Vec3DGrid>())  doProcessTypedGrid<Vec3DGrid>(grid, op);
    else return false;
    return true;
}

template<class TreeType>
class MinMaxVoxel
{
public:
    using LeafArray = openvdb::tree::LeafManager<TreeType>;
    using ValueType = typename TreeType::ValueType;

    // LeafArray = openvdb::tree::LeafManager<TreeType> leafs(myTree)
    MinMaxVoxel(LeafArray&);

    void runParallel();
    void runSerial();

    const ValueType& minVoxel() const { return mMin; }
    const ValueType& maxVoxel() const { return mMax; }

    inline MinMaxVoxel(const MinMaxVoxel<TreeType>&, tbb::split);
    inline void operator()(const tbb::blocked_range<size_t>&);
    inline void join(const MinMaxVoxel<TreeType>&);

private:
    LeafArray& mLeafArray;
    ValueType mMin, mMax;
};


template <class TreeType>
MinMaxVoxel<TreeType>::MinMaxVoxel(LeafArray& leafs)
    : mLeafArray(leafs)
    , mMin(std::numeric_limits<ValueType>::max())
    , mMax(-mMin)
{
}


template <class TreeType>
inline
MinMaxVoxel<TreeType>::MinMaxVoxel(const MinMaxVoxel<TreeType>& rhs, tbb::split)
    : mLeafArray(rhs.mLeafArray)
    , mMin(std::numeric_limits<ValueType>::max())
    , mMax(-mMin)
{
}


template <class TreeType>
void
MinMaxVoxel<TreeType>::runParallel()
{
    tbb::parallel_reduce(mLeafArray.getRange(), *this);
}


template <class TreeType>
void
MinMaxVoxel<TreeType>::runSerial()
{
    (*this)(mLeafArray.getRange());
}


template <class TreeType>
inline void
MinMaxVoxel<TreeType>::operator()(const tbb::blocked_range<size_t>& range)
{
    typename TreeType::LeafNodeType::ValueOnCIter iter;

    for (size_t n = range.begin(); n < range.end(); ++n) {
        iter = mLeafArray.leaf(n).cbeginValueOn();
        for (; iter; ++iter) {
            const ValueType value = iter.getValue();
            mMin = std::min(mMin, value);
            mMax = std::max(mMax, value);
        }
    }
}


template <class TreeType>
inline void
MinMaxVoxel<TreeType>::join(const MinMaxVoxel<TreeType>& rhs)
{
    mMin = std::min(mMin, rhs.mMin);
    mMax = std::max(mMax, rhs.mMax);
}

} // namespace util


////////////////////////////////////////


// BufferObject

BufferObject::BufferObject():
    mVertexBuffer(0),
    mNormalBuffer(0),
    mIndexBuffer(0),
    mColorBuffer(0),
    mPrimType(GL_POINTS),
    mPrimNum(0)
{
}

BufferObject::~BufferObject() { clear(); }

void
BufferObject::render() const
{
    if (mPrimNum == 0 || !glIsBuffer(mIndexBuffer) || !glIsBuffer(mVertexBuffer)) {
        OPENVDB_LOG_DEBUG_RUNTIME("request to render empty or uninitialized buffer");
        return;
    }

    const bool usesColorBuffer = glIsBuffer(mColorBuffer);
    const bool usesNormalBuffer = glIsBuffer(mNormalBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    if (usesColorBuffer) {
        glBindBuffer(GL_ARRAY_BUFFER, mColorBuffer);
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_FLOAT, 0, 0);
    }

    if (usesNormalBuffer) {
        glEnableClientState(GL_NORMAL_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
        glNormalPointer(GL_FLOAT, 0, 0);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
    glDrawElements(mPrimType, mPrimNum, GL_UNSIGNED_INT, 0);

    // disable client-side capabilities
    if (usesColorBuffer) glDisableClientState(GL_COLOR_ARRAY);
    if (usesNormalBuffer) glDisableClientState(GL_NORMAL_ARRAY);

    // release vbo's
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void
BufferObject::genIndexBuffer(const std::vector<GLuint>& v, GLenum primType)
{
    // clear old buffer
    if (glIsBuffer(mIndexBuffer) == GL_TRUE) glDeleteBuffers(1, &mIndexBuffer);

    // gen new buffer
    glGenBuffers(1, &mIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
    if (glIsBuffer(mIndexBuffer) == GL_FALSE) throw "Error: Unable to create index buffer";

    // upload data
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        sizeof(GLuint) * v.size(), &v[0], GL_STATIC_DRAW); // upload data
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to upload index buffer data";

    // release buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    mPrimNum = GLsizei(v.size());
    mPrimType = primType;
}

void
BufferObject::genVertexBuffer(const std::vector<GLfloat>& v)
{
    if (glIsBuffer(mVertexBuffer) == GL_TRUE) glDeleteBuffers(1, &mVertexBuffer);

    glGenBuffers(1, &mVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    if (glIsBuffer(mVertexBuffer) == GL_FALSE) throw "Error: Unable to create vertex buffer";

    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * v.size(), &v[0], GL_STATIC_DRAW);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to upload vertex buffer data";

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
BufferObject::genNormalBuffer(const std::vector<GLfloat>& v)
{
    if (glIsBuffer(mNormalBuffer) == GL_TRUE) glDeleteBuffers(1, &mNormalBuffer);

    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    if (glIsBuffer(mNormalBuffer) == GL_FALSE) throw "Error: Unable to create normal buffer";

    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * v.size(), &v[0], GL_STATIC_DRAW);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to upload normal buffer data";

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
BufferObject::genColorBuffer(const std::vector<GLfloat>& v)
{
    if (glIsBuffer(mColorBuffer) == GL_TRUE) glDeleteBuffers(1, &mColorBuffer);

    glGenBuffers(1, &mColorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mColorBuffer);
    if (glIsBuffer(mColorBuffer) == GL_FALSE) throw "Error: Unable to create color buffer";

    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * v.size(), &v[0], GL_STATIC_DRAW);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to upload color buffer data";

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
BufferObject::clear()
{
    if (glIsBuffer(mIndexBuffer) == GL_TRUE) glDeleteBuffers(1, &mIndexBuffer);
    if (glIsBuffer(mVertexBuffer) == GL_TRUE) glDeleteBuffers(1, &mVertexBuffer);
    if (glIsBuffer(mColorBuffer) == GL_TRUE) glDeleteBuffers(1, &mColorBuffer);
    if (glIsBuffer(mNormalBuffer) == GL_TRUE) glDeleteBuffers(1, &mNormalBuffer);

    mPrimType = GL_POINTS;
    mPrimNum = 0;
}


////////////////////////////////////////


ShaderProgram::ShaderProgram():
    mProgram(0),
    mVertShader(0),
    mFragShader(0)
{
}

ShaderProgram::~ShaderProgram() { clear(); }

void
ShaderProgram::setVertShader(const std::string& s)
{
    mVertShader = glCreateShader(GL_VERTEX_SHADER);
    if (glIsShader(mVertShader) == GL_FALSE) throw "Error: Unable to create shader program.";

    GLint length = GLint(s.length());
    const char *str = s.c_str();
    glShaderSource(mVertShader, 1, &str, &length);

    glCompileShader(mVertShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to compile vertex shader.";
}

void
ShaderProgram::setFragShader(const std::string& s)
{
    mFragShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (glIsShader(mFragShader) == GL_FALSE) throw "Error: Unable to create shader program.";

    GLint length = GLint(s.length());
    const char *str = s.c_str();
    glShaderSource(mFragShader, 1, &str, &length);

    glCompileShader(mFragShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to compile fragment shader.";
}

void
ShaderProgram::build()
{
    mProgram = glCreateProgram();
    if (glIsProgram(mProgram) == GL_FALSE) throw "Error: Unable to create shader program.";

    if (glIsShader(mVertShader) == GL_TRUE) glAttachShader(mProgram, mVertShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to attach vertex shader.";

    if (glIsShader(mFragShader) == GL_TRUE) glAttachShader(mProgram, mFragShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to attach fragment shader.";


    glLinkProgram(mProgram);

    GLint linked = 0;
    glGetProgramiv(mProgram, GL_LINK_STATUS, &linked);

    if (!linked) throw "Error: Unable to link shader program.";
}

void
ShaderProgram::build(const std::vector<GLchar*>& attributes)
{
    mProgram = glCreateProgram();
    if (glIsProgram(mProgram) == GL_FALSE) throw "Error: Unable to create shader program.";

    for (GLuint n = 0, N = GLuint(attributes.size()); n < N; ++n) {
        glBindAttribLocation(mProgram, n, attributes[n]);
    }

    if (glIsShader(mVertShader) == GL_TRUE) glAttachShader(mProgram, mVertShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to attach vertex shader.";

    if (glIsShader(mFragShader) == GL_TRUE) glAttachShader(mProgram, mFragShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to attach fragment shader.";

    glLinkProgram(mProgram);

    GLint linked;
    glGetProgramiv(mProgram, GL_LINK_STATUS, &linked);

    if (!linked) throw "Error: Unable to link shader program.";
}

void
ShaderProgram::startShading() const
{
    if (glIsProgram(mProgram) == GL_FALSE) {
        throw "Error: called startShading() on uncompiled shader program.";
    }
    glUseProgram(mProgram);
}

void
ShaderProgram::stopShading() const
{
    glUseProgram(0);
}

void
ShaderProgram::clear()
{
    GLsizei numShaders = 0;
    GLuint shaders[2] = { 0, 0 };

    glGetAttachedShaders(mProgram, 2, &numShaders, shaders);

    // detach and remove shaders
    for (GLsizei n = 0; n < numShaders; ++n) {

        glDetachShader(mProgram, shaders[n]);

        if (glIsShader(shaders[n]) == GL_TRUE) glDeleteShader(shaders[n]);
    }

    // remove program
    if (glIsProgram(mProgram)) glDeleteProgram(mProgram);
}


////////////////////////////////////////

// ViewportModule

ViewportModule::ViewportModule():
    mAxisGnomonScale(1.5),
    mGroundPlaneScale(8.0)
{
}


void
ViewportModule::render()
{
    if (!mIsVisible) return;

    /// @todo use VBO's

    // Ground plane
    glPushMatrix();
    glScalef(mGroundPlaneScale, mGroundPlaneScale, mGroundPlaneScale);
    glColor3d(0.6, 0.6, 0.6);

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN

    float step = 0.125;
    for (float x = -1; x < 1.125; x+=step) {

        if (std::fabs(x) == 0.5 || std::fabs(x) == 0.0) {
            glLineWidth(1.5);
        } else {
            glLineWidth(1.0);
        }

        glBegin(GL_LINES);
        glVertex3f(x, 0, 1);
        glVertex3f(x, 0, -1);
        glVertex3f(1, 0, x);
        glVertex3f(-1, 0, x);
        glEnd();
    }

    OPENVDB_NO_FP_EQUALITY_WARNING_END


    glPopMatrix();

    // Axis gnomon
    GLfloat modelview[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, &modelview[0]);

    // Stash current viewport settigs.
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, &viewport[0]);

    GLint width = viewport[2] / 20;
    GLint height = viewport[3] / 20;
    glViewport(0, 0, width, height);


    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();


    GLfloat campos[3] = { modelview[2], modelview[6], modelview[10] };
    GLfloat up[3] = { modelview[1], modelview[5], modelview[9] };

    gluLookAt(campos[0], campos[1], campos[2], 0.0, 0.0, 0.0, up[0], up[1], up[2]);

    glScalef(mAxisGnomonScale, mAxisGnomonScale, mAxisGnomonScale);

    glLineWidth(1.0);

    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(1, 0, 0);

    glColor3f(0.0f, 1.0f, 0.0f );
    glVertex3f(0, 0, 0);
    glVertex3f(0, 1, 0);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 1);
    glEnd();

    glLineWidth(1.0);

    // reset viewport
    glPopMatrix();
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

}


////////////////////////////////////////


class TreeTopologyOp
{
public:
    TreeTopologyOp(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        using openvdb::Index64;

        Index64 nodeCount = grid->tree().leafCount() + grid->tree().nonLeafCount();
        const Index64 N = nodeCount * 8 * 3;

        std::vector<GLfloat> points(N);
        std::vector<GLfloat> colors(N);
        std::vector<GLuint> indices(N);


        openvdb::Vec3d ptn;
        openvdb::Vec3s color;
        openvdb::CoordBBox bbox;
        Index64 pOffset = 0, iOffset = 0,  cOffset = 0, idx = 0;

        for (typename GridType::TreeType::NodeCIter iter = grid->tree().cbeginNode(); iter; ++iter)
        {
            iter.getBoundingBox(bbox);

            // Nodes are rendered as cell-centered
            const openvdb::Vec3d min(bbox.min().x()-0.5, bbox.min().y()-0.5, bbox.min().z()-0.5);
            const openvdb::Vec3d max(bbox.max().x()+0.5, bbox.max().y()+0.5, bbox.max().z()+0.5);

            // corner 1
            ptn = grid->indexToWorld(min);
            points[pOffset++] = static_cast<GLfloat>(ptn[0]);
            points[pOffset++] = static_cast<GLfloat>(ptn[1]);
            points[pOffset++] = static_cast<GLfloat>(ptn[2]);

            // corner 2
            ptn = openvdb::Vec3d(min.x(), min.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = static_cast<GLfloat>(ptn[0]);
            points[pOffset++] = static_cast<GLfloat>(ptn[1]);
            points[pOffset++] = static_cast<GLfloat>(ptn[2]);

            // corner 3
            ptn = openvdb::Vec3d(max.x(), min.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = static_cast<GLfloat>(ptn[0]);
            points[pOffset++] = static_cast<GLfloat>(ptn[1]);
            points[pOffset++] = static_cast<GLfloat>(ptn[2]);

            // corner 4
            ptn = openvdb::Vec3d(max.x(), min.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = static_cast<GLfloat>(ptn[0]);
            points[pOffset++] = static_cast<GLfloat>(ptn[1]);
            points[pOffset++] = static_cast<GLfloat>(ptn[2]);

            // corner 5
            ptn = openvdb::Vec3d(min.x(), max.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = static_cast<GLfloat>(ptn[0]);
            points[pOffset++] = static_cast<GLfloat>(ptn[1]);
            points[pOffset++] = static_cast<GLfloat>(ptn[2]);

            // corner 6
            ptn = openvdb::Vec3d(min.x(), max.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = static_cast<GLfloat>(ptn[0]);
            points[pOffset++] = static_cast<GLfloat>(ptn[1]);
            points[pOffset++] = static_cast<GLfloat>(ptn[2]);

            // corner 7
            ptn = grid->indexToWorld(max);
            points[pOffset++] = static_cast<GLfloat>(ptn[0]);
            points[pOffset++] = static_cast<GLfloat>(ptn[1]);
            points[pOffset++] = static_cast<GLfloat>(ptn[2]);

            // corner 8
            ptn = openvdb::Vec3d(max.x(), max.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = static_cast<GLfloat>(ptn[0]);
            points[pOffset++] = static_cast<GLfloat>(ptn[1]);
            points[pOffset++] = static_cast<GLfloat>(ptn[2]);


            // edge 1
            indices[iOffset++] = GLuint(idx);
            indices[iOffset++] = GLuint(idx + 1);
            // edge 2
            indices[iOffset++] = GLuint(idx + 1);
            indices[iOffset++] = GLuint(idx + 2);
            // edge 3
            indices[iOffset++] = GLuint(idx + 2);
            indices[iOffset++] = GLuint(idx + 3);
            // edge 4
            indices[iOffset++] = GLuint(idx + 3);
            indices[iOffset++] = GLuint(idx);
            // edge 5
            indices[iOffset++] = GLuint(idx + 4);
            indices[iOffset++] = GLuint(idx + 5);
            // edge 6
            indices[iOffset++] = GLuint(idx + 5);
            indices[iOffset++] = GLuint(idx + 6);
            // edge 7
            indices[iOffset++] = GLuint(idx + 6);
            indices[iOffset++] = GLuint(idx + 7);
            // edge 8
            indices[iOffset++] = GLuint(idx + 7);
            indices[iOffset++] = GLuint(idx + 4);
            // edge 9
            indices[iOffset++] = GLuint(idx);
            indices[iOffset++] = GLuint(idx + 4);
            // edge 10
            indices[iOffset++] = GLuint(idx + 1);
            indices[iOffset++] = GLuint(idx + 5);
            // edge 11
            indices[iOffset++] = GLuint(idx + 2);
            indices[iOffset++] = GLuint(idx + 6);
            // edge 12
            indices[iOffset++] = GLuint(idx + 3);
            indices[iOffset++] = GLuint(idx + 7);

            // node vertex color
            const int level = iter.getLevel();
            color = sNodeColors[(level == 0) ? 3 : (level == 1) ? 2 : 1];

            for (Index64 n = 0; n < 8; ++n) {
                colors[cOffset++] = color[0];
                colors[cOffset++] = color[1];
                colors[cOffset++] = color[2];
            }

            idx += 8;
        } // end node iteration

        // gen buffers and upload data to GPU
        mBuffer->genVertexBuffer(points);
        mBuffer->genColorBuffer(colors);
        mBuffer->genIndexBuffer(indices, GL_LINES);
    }

private:
    BufferObject *mBuffer;

    static openvdb::Vec3s sNodeColors[];
}; // TreeTopologyOp


openvdb::Vec3s TreeTopologyOp::sNodeColors[] = {
    openvdb::Vec3s(0.045f, 0.045f, 0.045f),         // root
    openvdb::Vec3s(0.0432f, 0.33f, 0.0411023f),     // first internal node level
    openvdb::Vec3s(0.871f, 0.394f, 0.01916f),       // intermediate internal node levels
    openvdb::Vec3s(0.00608299f, 0.279541f, 0.625f)  // leaf nodes
};


////////////////////////////////////////

// Tree topology render module

TreeTopologyModule::TreeTopologyModule(const openvdb::GridBase::ConstPtr& grid):
    mGrid(grid),
    mIsInitialized(false)
{
    mShader.setVertShader(
        "#version 120\n"
        "void main() {\n"
        "gl_FrontColor = gl_Color;\n"
        "gl_Position =  ftransform();\n"
        "gl_ClipVertex = gl_ModelViewMatrix * gl_Vertex;\n"
        "}\n");

    mShader.setFragShader(
        "#version 120\n"
         "void main() {\n"
            "gl_FragColor = gl_Color;}\n");

    mShader.build();
}


void
TreeTopologyModule::init()
{
    mIsInitialized = true;

    // extract grid topology
    TreeTopologyOp drawTopology(mBufferObject);

    if (!util::processTypedGrid(mGrid, drawTopology)) {
        OPENVDB_LOG_INFO("Ignoring unrecognized grid type"
            " during tree topology module initialization.");
    }
}


void
TreeTopologyModule::render()
{
    if (!mIsVisible) return;
    if (!mIsInitialized) init();

    mShader.startShading();

    mBufferObject.render();

    mShader.stopShading();
}


////////////////////////////////////////


template<typename TreeType>
class PointGenerator
{
public:
    using LeafManagerType = openvdb::tree::LeafManager<TreeType>;

    PointGenerator(
        std::vector<GLfloat>& points,
        std::vector<GLuint>& indices,
        LeafManagerType& leafs,
        std::vector<size_t>& indexMap,
        const openvdb::math::Transform& transform,
        openvdb::Index64 voxelsPerLeaf = TreeType::LeafNodeType::NUM_VOXELS)
        : mPoints(points)
        , mIndices(indices)
        , mLeafs(leafs)
        , mIndexMap(indexMap)
        , mTransform(transform)
        , mVoxelsPerLeaf(voxelsPerLeaf)
    {
    }

    void runParallel()
    {
        tbb::parallel_for(mLeafs.getRange(), *this);
    }


    inline void operator()(const typename LeafManagerType::RangeType& range) const
    {
        using openvdb::Index64;

        using ValueOnCIter = typename TreeType::LeafNodeType::ValueOnCIter;

        openvdb::Vec3d pos;
        size_t index = 0;
        Index64 activeVoxels = 0;

        for (size_t n = range.begin(); n < range.end(); ++n) {

            index = mIndexMap[n];
            ValueOnCIter it = mLeafs.leaf(n).cbeginValueOn();

            activeVoxels = mLeafs.leaf(n).onVoxelCount();

            if (activeVoxels <= mVoxelsPerLeaf) {

                for ( ; it; ++it) {
                    pos = mTransform.indexToWorld(it.getCoord());
                    insertPoint(pos, index);
                    ++index;
                }

            } else if (1 == mVoxelsPerLeaf) {

                 pos = mTransform.indexToWorld(it.getCoord());
                 insertPoint(pos, index);

            } else {

                std::vector<openvdb::Coord> coords;
                coords.reserve(static_cast<size_t>(activeVoxels));
                for ( ; it; ++it) { coords.push_back(it.getCoord()); }

                pos = mTransform.indexToWorld(coords[0]);
                insertPoint(pos, index);
                ++index;

                pos = mTransform.indexToWorld(coords[static_cast<size_t>(activeVoxels-1)]);
                insertPoint(pos, index);
                ++index;

                Index64 r = Index64(std::floor(double(mVoxelsPerLeaf) / activeVoxels));
                for (Index64 i = 1, I = mVoxelsPerLeaf - 2; i < I; ++i) {
                    pos = mTransform.indexToWorld(coords[static_cast<size_t>(i * r)]);
                    insertPoint(pos, index);
                    ++index;
                }
            }
        }
    }

private:
    void insertPoint(const openvdb::Vec3d& pos, size_t index) const
    {
        mIndices[index] = GLuint(index);
        const size_t element = index * 3;
        mPoints[element    ] = static_cast<GLfloat>(pos[0]);
        mPoints[element + 1] = static_cast<GLfloat>(pos[1]);
        mPoints[element + 2] = static_cast<GLfloat>(pos[2]);
    }

    std::vector<GLfloat>& mPoints;
    std::vector<GLuint>& mIndices;
    LeafManagerType& mLeafs;
    std::vector<size_t>& mIndexMap;
    const openvdb::math::Transform& mTransform;
    const openvdb::Index64 mVoxelsPerLeaf;
}; // PointGenerator


template<typename GridType>
class NormalGenerator
{
public:
    using AccessorType = typename GridType::ConstAccessor;
    using Grad = openvdb::math::ISGradient<openvdb::math::CD_2ND>;

    NormalGenerator(const AccessorType& acc): mAccessor(acc) {}

    NormalGenerator(const NormalGenerator&) = delete;
    NormalGenerator& operator=(const NormalGenerator&) = delete;

    void operator()(const openvdb::Coord& ijk, openvdb::Vec3d& normal)
    {
        openvdb::Vec3d v{Grad::result(mAccessor, ijk)};
        const double length = v.length();
        if (length > 1.0e-7) {
            v *= 1.0 / length;
            normal = v;
        }
    }

private:
    const AccessorType& mAccessor;
}; // class NormalGenerator

// Specialization for PointDataGrids, for which normals are not generated
template<>
class NormalGenerator<openvdb::points::PointDataGrid>
{
public:
    NormalGenerator(const openvdb::points::PointDataGrid::ConstAccessor&) {}
    NormalGenerator(const NormalGenerator&) = delete;
    NormalGenerator& operator=(const NormalGenerator&) = delete;
    void operator()(const openvdb::Coord&, openvdb::Vec3d&) {}
};


template<typename GridType>
class PointAttributeGenerator
{
public:
    using ValueType = typename GridType::ValueType;

    PointAttributeGenerator(
        std::vector<GLfloat>& points,
        std::vector<GLfloat>& colors,
        const GridType& grid,
        ValueType minValue,
        ValueType maxValue,
        openvdb::Vec3s (&colorMap)[4],
        bool isLevelSet = false)
        : mPoints(points)
        , mColors(colors)
        , mNormals(nullptr)
        , mGrid(grid)
        , mAccessor(grid.tree())
        , mMinValue(minValue)
        , mMaxValue(maxValue)
        , mColorMap(colorMap)
        , mIsLevelSet(isLevelSet)
        , mZeroValue(openvdb::zeroVal<ValueType>())
    {
        init();
    }

    PointAttributeGenerator(
        std::vector<GLfloat>& points,
        std::vector<GLfloat>& colors,
        std::vector<GLfloat>& normals,
        const GridType& grid,
        ValueType minValue,
        ValueType maxValue,
        openvdb::Vec3s (&colorMap)[4],
        bool isLevelSet = false)
        : mPoints(points)
        , mColors(colors)
        , mNormals(&normals)
        , mGrid(grid)
        , mAccessor(grid.tree())
        , mMinValue(minValue)
        , mMaxValue(maxValue)
        , mColorMap(colorMap)
        , mIsLevelSet(isLevelSet)
        , mZeroValue(openvdb::zeroVal<ValueType>())
    {
        init();
    }

    void runParallel()
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, (mPoints.size() / 3)), *this);
    }

    inline void operator()(const tbb::blocked_range<size_t>& range) const
    {
        openvdb::Coord ijk;
        openvdb::Vec3d pos, normal(0.0, -1.0, 0.0);
        openvdb::Vec3s color(0.9f, 0.3f, 0.3f);
        float w = 0.0;
        NormalGenerator<GridType> computeNormal{mAccessor};

        size_t e1, e2, e3, voxelNum = 0;
        for (size_t n = range.begin(); n < range.end(); ++n) {
            e1 = 3 * n;
            e2 = e1 + 1;
            e3 = e2 + 1;

            pos[0] = mPoints[e1];
            pos[1] = mPoints[e2];
            pos[2] = mPoints[e3];

            pos = mGrid.worldToIndex(pos);
            ijk[0] = int(pos[0]);
            ijk[1] = int(pos[1]);
            ijk[2] = int(pos[2]);

            const ValueType& value = mAccessor.getValue(ijk);

            if (value < mZeroValue) { // is negative
                if (mIsLevelSet) {
                    color = mColorMap[1];
                } else {
                    w = (float(value) - mOffset[1]) * mScale[1];
                    color = openvdb::Vec3s(w * mColorMap[0] + (1.0 - w) * mColorMap[1]);
                }
            } else {
                if (mIsLevelSet) {
                    color = mColorMap[2];
                } else {
                    w = (float(value) - mOffset[0]) * mScale[0];
                    color = openvdb::Vec3s(w * mColorMap[2] + (1.0 - w) * mColorMap[3]);
                }
            }

            mColors[e1] = color[0];
            mColors[e2] = color[1];
            mColors[e3] = color[2];

            if (mNormals) {
                if ((voxelNum % 2) == 0) { computeNormal(ijk, normal); }
                ++voxelNum;
                (*mNormals)[e1] = static_cast<GLfloat>(normal[0]);
                (*mNormals)[e2] = static_cast<GLfloat>(normal[1]);
                (*mNormals)[e3] = static_cast<GLfloat>(normal[2]);
            }
        }
    }

private:

    void init()
    {
        mOffset[0] = static_cast<float>(std::min(mZeroValue, mMinValue));
        mScale[0] = static_cast<float>(
            1.0 / (std::abs(std::max(mZeroValue, mMaxValue) - mOffset[0])));
        mOffset[1] = static_cast<float>(std::min(mZeroValue, mMinValue));
        mScale[1] = static_cast<float>(
            1.0 / (std::abs(std::max(mZeroValue, mMaxValue) - mOffset[1])));
    }

    std::vector<GLfloat>& mPoints;
    std::vector<GLfloat>& mColors;
    std::vector<GLfloat>* mNormals;

    const GridType& mGrid;
    openvdb::tree::ValueAccessor<const typename GridType::TreeType> mAccessor;

    ValueType mMinValue, mMaxValue;
    openvdb::Vec3s (&mColorMap)[4];
    const bool mIsLevelSet;

    ValueType mZeroValue;
    float mOffset[2], mScale[2];
}; // PointAttributeGenerator


////////////////////////////////////////


class ActiveScalarValuesOp
{
public:
    ActiveScalarValuesOp(
        BufferObject& interiorBuffer, BufferObject& surfaceBuffer)
        : mInteriorBuffer(&interiorBuffer)
        , mSurfaceBuffer(&surfaceBuffer)
    {
    }

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        using openvdb::Index64;

        const Index64 maxVoxelPoints = 26000000;

        openvdb::Vec3s colorMap[4];
        colorMap[0] = openvdb::Vec3s(0.3f, 0.9f, 0.3f); // green
        colorMap[1] = openvdb::Vec3s(0.9f, 0.3f, 0.3f); // red
        colorMap[2] = openvdb::Vec3s(0.9f, 0.9f, 0.3f); // yellow
        colorMap[3] = openvdb::Vec3s(0.3f, 0.3f, 0.9f); // blue

        //////////

        using ValueType = typename GridType::ValueType;
        using TreeType = typename GridType::TreeType;
        using BoolTreeT = typename TreeType::template ValueConverter<bool>::Type;

        const TreeType& tree = grid->tree();
        const bool isLevelSetGrid = grid->getGridClass() == openvdb::GRID_LEVEL_SET;

        ValueType minValue, maxValue;
        openvdb::tree::LeafManager<const TreeType> leafs(tree);

        {
            util::MinMaxVoxel<const TreeType> minmax(leafs);
            minmax.runParallel();
            minValue = minmax.minVoxel();
            maxValue = minmax.maxVoxel();
        }

        openvdb::Index64 voxelsPerLeaf = TreeType::LeafNodeType::NUM_VOXELS;

        if (!isLevelSetGrid) {

            typename BoolTreeT::Ptr interiorMask(new BoolTreeT(false));

            { // Generate Interior Points
                interiorMask->topologyUnion(tree);
                interiorMask->voxelizeActiveTiles();

                if (interiorMask->activeLeafVoxelCount() > maxVoxelPoints) {
                    voxelsPerLeaf = std::max<Index64>(1,
                        (maxVoxelPoints / interiorMask->leafCount()));
                }

                openvdb::tools::erodeVoxels(*interiorMask, 2);

                openvdb::tree::LeafManager<BoolTreeT> maskleafs(*interiorMask);
                std::vector<size_t> indexMap(maskleafs.leafCount());
                size_t voxelCount = 0;
                for (Index64 l = 0, L = maskleafs.leafCount(); l < L; ++l) {
                    indexMap[l] = voxelCount;
                    voxelCount += std::min(maskleafs.leaf(l).onVoxelCount(), voxelsPerLeaf);
                }

                std::vector<GLfloat> points(voxelCount * 3), colors(voxelCount * 3);
                std::vector<GLuint> indices(voxelCount);

                PointGenerator<BoolTreeT> pointGen(
                    points, indices, maskleafs, indexMap, grid->transform(), voxelsPerLeaf);
                pointGen.runParallel();


                PointAttributeGenerator<GridType> attributeGen(
                    points, colors, *grid, minValue, maxValue, colorMap);
                attributeGen.runParallel();


                // gen buffers and upload data to GPU
                mInteriorBuffer->genVertexBuffer(points);
                mInteriorBuffer->genColorBuffer(colors);
                mInteriorBuffer->genIndexBuffer(indices, GL_POINTS);
            }

            { // Generate Surface Points
                typename BoolTreeT::Ptr surfaceMask(new BoolTreeT(false));
                surfaceMask->topologyUnion(tree);
                surfaceMask->voxelizeActiveTiles();

                openvdb::tree::ValueAccessor<BoolTreeT> interiorAcc(*interiorMask);
                for (typename BoolTreeT::LeafIter leafIt = surfaceMask->beginLeaf();
                    leafIt; ++leafIt)
                {
                    const typename BoolTreeT::LeafNodeType* leaf =
                        interiorAcc.probeConstLeaf(leafIt->origin());
                    if (leaf) leafIt->topologyDifference(*leaf, false);
                }
                openvdb::tools::pruneInactive(*surfaceMask);

                openvdb::tree::LeafManager<BoolTreeT> maskleafs(*surfaceMask);
                std::vector<size_t> indexMap(maskleafs.leafCount());
                size_t voxelCount = 0;
                for (Index64 l = 0, L = maskleafs.leafCount(); l < L; ++l) {
                    indexMap[l] = voxelCount;
                    voxelCount += std::min(maskleafs.leaf(l).onVoxelCount(), voxelsPerLeaf);
                }

                std::vector<GLfloat>
                    points(voxelCount * 3),
                    colors(voxelCount * 3),
                    normals(voxelCount * 3);
                std::vector<GLuint> indices(voxelCount);

                PointGenerator<BoolTreeT> pointGen(
                    points, indices, maskleafs, indexMap, grid->transform(), voxelsPerLeaf);
                pointGen.runParallel();

                PointAttributeGenerator<GridType> attributeGen(
                    points, colors, normals, *grid, minValue, maxValue, colorMap);
                attributeGen.runParallel();

                mSurfaceBuffer->genVertexBuffer(points);
                mSurfaceBuffer->genColorBuffer(colors);
                mSurfaceBuffer->genNormalBuffer(normals);
                mSurfaceBuffer->genIndexBuffer(indices, GL_POINTS);
            }

            return;
        }

        // Level set rendering
        if (tree.activeLeafVoxelCount() > maxVoxelPoints) {
            voxelsPerLeaf = std::max<Index64>(1, (maxVoxelPoints / tree.leafCount()));
        }

        std::vector<size_t> indexMap(leafs.leafCount());
        size_t voxelCount = 0;
        for (Index64 l = 0, L = leafs.leafCount(); l < L; ++l) {
            indexMap[l] = voxelCount;
            voxelCount += std::min(leafs.leaf(l).onVoxelCount(), voxelsPerLeaf);
        }

        std::vector<GLfloat>
            points(voxelCount * 3),
            colors(voxelCount * 3),
            normals(voxelCount * 3);
        std::vector<GLuint> indices(voxelCount);

        PointGenerator<const TreeType> pointGen(
            points, indices, leafs, indexMap, grid->transform(), voxelsPerLeaf);
        pointGen.runParallel();

        PointAttributeGenerator<GridType> attributeGen(
            points, colors, normals, *grid, minValue, maxValue, colorMap, isLevelSetGrid);
        attributeGen.runParallel();

        mSurfaceBuffer->genVertexBuffer(points);
        mSurfaceBuffer->genColorBuffer(colors);
        mSurfaceBuffer->genNormalBuffer(normals);
        mSurfaceBuffer->genIndexBuffer(indices, GL_POINTS);
    }

private:
    BufferObject *mInteriorBuffer;
    BufferObject *mSurfaceBuffer;
}; // ActiveScalarValuesOp


class ActiveVectorValuesOp
{
public:
    ActiveVectorValuesOp(BufferObject& vectorBuffer)
        : mVectorBuffer(&vectorBuffer)
    {
    }

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        using openvdb::Index64;

        using ValueType = typename GridType::ValueType;
        using TreeType = typename GridType::TreeType;
        using BoolTreeT = typename TreeType::template ValueConverter<bool>::Type;


        const TreeType& tree = grid->tree();

        double length = 0.0;
        {
            ValueType minVal, maxVal;
            tree.evalMinMax(minVal, maxVal);
            length = maxVal.length();
        }

        typename BoolTreeT::Ptr mask(new BoolTreeT(false));
        mask->topologyUnion(tree);
        mask->voxelizeActiveTiles();

        ///@todo thread and restructure.

        const Index64 voxelCount = mask->activeLeafVoxelCount();

        const Index64 pointCount = voxelCount * 2;
        std::vector<GLfloat> points(pointCount*3), colors(pointCount*3);
        std::vector<GLuint> indices(pointCount);

        openvdb::Coord ijk;
        openvdb::Vec3d pos, color, normal;
        openvdb::tree::LeafManager<BoolTreeT> leafs(*mask);

        openvdb::tree::ValueAccessor<const TreeType> acc(tree);

        Index64 idx = 0, pt = 0, cc = 0;
        for (Index64 l = 0, L = leafs.leafCount(); l < L; ++l) {
            typename BoolTreeT::LeafNodeType::ValueOnIter iter = leafs.leaf(l).beginValueOn();
            for (; iter; ++iter) {
                ijk = iter.getCoord();
                ValueType vec = acc.getValue(ijk);

                pos = grid->indexToWorld(ijk);

                points[idx++] = static_cast<GLfloat>(pos[0]);
                points[idx++] = static_cast<GLfloat>(pos[1]);
                points[idx++] = static_cast<GLfloat>(pos[2]);

                indices[pt] = GLuint(pt);
                ++pt;
                indices[pt] = GLuint(pt);

                ++pt;
                double w = vec.length() / length;
                vec.normalize();
                pos += grid->voxelSize()[0] * 0.9 * vec;

                points[idx++] = static_cast<GLfloat>(pos[0]);
                points[idx++] = static_cast<GLfloat>(pos[1]);
                points[idx++] = static_cast<GLfloat>(pos[2]);


                color = w * openvdb::Vec3d(0.9, 0.3, 0.3)
                    + (1.0 - w) * openvdb::Vec3d(0.3, 0.3, 0.9);

                colors[cc++] = static_cast<GLfloat>(color[0] * 0.3);
                colors[cc++] = static_cast<GLfloat>(color[1] * 0.3);
                colors[cc++] = static_cast<GLfloat>(color[2] * 0.3);

                colors[cc++] = static_cast<GLfloat>(color[0]);
                colors[cc++] = static_cast<GLfloat>(color[1]);
                colors[cc++] = static_cast<GLfloat>(color[2]);
            }
        }

        mVectorBuffer->genVertexBuffer(points);
        mVectorBuffer->genColorBuffer(colors);
        mVectorBuffer->genIndexBuffer(indices, GL_LINES);
    }

private:
    BufferObject *mVectorBuffer;

}; // ActiveVectorValuesOp


class PointDataOp
{
public:
    using GLfloatVec = std::vector<GLfloat>;
    using GLuintVec = std::vector<GLuint>;

private:
    struct VectorAttributeWrapper
    {
        using ValueType = openvdb::Vec3f;

        struct Handle
        {
            explicit Handle(VectorAttributeWrapper& attribute):
                mValues(attribute.mValues), mIndices(attribute.mIndices) {}

            void set(openvdb::Index offset, openvdb::Index/*unused*/, const ValueType& value)
            {
                if (mIndices) (*mIndices)[offset] = static_cast<GLuint>(offset);
                offset *= 3;
                for (int i = 0; i < 3; ++i, ++offset) { mValues[offset] = value[i]; }
            }
        private:
            GLfloatVec& mValues;
            GLuintVec* mIndices;
        }; // struct Handle

        explicit VectorAttributeWrapper(GLfloatVec& values, GLuintVec* indices = nullptr):
            mValues(values), mIndices(indices) {}

        void expand() {}
        void compact() {}
    private:
        GLfloatVec& mValues;
        GLuintVec* mIndices;
    }; // struct VectorAttributeWrapper

public:
    explicit PointDataOp(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        const typename GridType::TreeType& tree = grid->tree();

        // obtain cumulative point offsets and total points
        std::vector<openvdb::Index64> pointOffsets;
        const openvdb::Index64 total = openvdb::points::pointOffsets(pointOffsets, tree);

        // @todo use glDrawArrays with GL_POINTS to avoid generating indices
        GLfloatVec values(total * 3);
        GLuintVec indices(total);

        VectorAttributeWrapper positionWrapper{values, &indices};
        openvdb::points::convertPointDataGridPosition(positionWrapper, *grid, pointOffsets, 0);

        // gen buffers and upload data to GPU
        mBuffer->genVertexBuffer(values);
        mBuffer->genIndexBuffer(indices, GL_POINTS);

        const auto leafIter = tree.cbeginLeaf();
        if (!leafIter) return;

        const size_t colorIdx = leafIter->attributeSet().find("Cd");
        if (colorIdx == openvdb::points::AttributeSet::INVALID_POS) return;

        const auto& colorArray = leafIter->constAttributeArray(colorIdx);
        if (colorArray.template hasValueType<openvdb::Vec3f>()) {
            VectorAttributeWrapper colorWrapper{values};
            openvdb::points::convertPointDataGridAttribute(colorWrapper, tree, pointOffsets,
                /*startOffset=*/0, static_cast<unsigned>(colorIdx));

            // gen color buffer
            mBuffer->genColorBuffer(values);
        }
    }

private:
    BufferObject* mBuffer;
}; // PointDataOp


////////////////////////////////////////

// Active value render module

VoxelModule::VoxelModule(const openvdb::GridBase::ConstPtr& grid):
    mGrid(grid),
    mIsInitialized(false)
{
    mFlatShader.setVertShader(
        "#version 120\n"
        "void main() {\n"
        "gl_FrontColor = gl_Color;\n"
        "gl_Position =  ftransform();\n"
        "gl_ClipVertex = gl_ModelViewMatrix * gl_Vertex;\n"
        "}\n");

    mFlatShader.setFragShader(
        "#version 120\n"
         "void main() {\n"
            "gl_FragColor = gl_Color;}\n");

    mFlatShader.build();

    mSurfaceShader.setVertShader(
        "#version 120\n"
        "varying vec3 normal;\n"
        "void main() {\n"
            "gl_FrontColor = gl_Color;\n"
            "normal = normalize(gl_NormalMatrix * gl_Normal);\n"
            "gl_Position =  ftransform();\n"
            "gl_ClipVertex = gl_ModelViewMatrix * gl_Vertex;\n"
        "}\n");


    mSurfaceShader.setFragShader(
        "#version 120\n"
        "varying vec3 normal;\n"
        "void main() {\n"
            "vec3 normalized_normal = normalize(normal);\n"
            "float w = 0.5 * (1.0 + dot(normalized_normal, vec3(0.0, 1.0, 0.0)));\n"
            "vec4 diffuseColor = w * gl_Color + (1.0 - w) * (gl_Color * 0.3);\n"
            "gl_FragColor = diffuseColor;\n"
        "}\n");

    mSurfaceShader.build();
}


void
VoxelModule::init()
{
    mIsInitialized = true;

    if (mGrid->isType<openvdb::points::PointDataGrid>()) {
        mSurfaceBuffer.clear();
        PointDataOp drawPoints(mInteriorBuffer);
        util::doProcessTypedGrid<openvdb::points::PointDataGrid>(mGrid, drawPoints);
    } else {
        ActiveScalarValuesOp drawScalars(mInteriorBuffer, mSurfaceBuffer);
        if (!util::processTypedScalarOrPointDataGrid(mGrid, drawScalars)) {
            ActiveVectorValuesOp drawVectors(mVectorBuffer);
            if (!util::processTypedVectorGrid(mGrid, drawVectors)) {
                OPENVDB_LOG_INFO("Ignoring unrecognized grid type " << mGrid->type()
                    << " during active value module initialization.");
            }
        }
    }
}


void
VoxelModule::render()
{
    if (!mIsVisible) return;
    if (!mIsInitialized) init();

    mFlatShader.startShading();
        mInteriorBuffer.render();
        mVectorBuffer.render();
    mFlatShader.stopShading();

    mSurfaceShader.startShading();
        mSurfaceBuffer.render();
    mSurfaceShader.stopShading();
}


////////////////////////////////////////


class MeshOp
{
public:
    MeshOp(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        using openvdb::Index64;

        openvdb::tools::VolumeToMesh mesher(
            grid->getGridClass() == openvdb::GRID_LEVEL_SET ? 0.0 : 0.01);
        mesher(*grid);

        // Copy points and generate point normals.
        std::vector<GLfloat> points(mesher.pointListSize() * 3);
        std::vector<GLfloat> normals(mesher.pointListSize() * 3);

        openvdb::tree::ValueAccessor<const typename GridType::TreeType> acc(grid->tree());
        openvdb::math::GenericMap map(grid->transform());
        openvdb::Coord ijk;

        for (Index64 n = 0, i = 0,  N = mesher.pointListSize(); n < N; ++n) {
            const openvdb::Vec3s& p = mesher.pointList()[n];
            points[i++] = p[0];
            points[i++] = p[1];
            points[i++] = p[2];
        }

        // Copy primitives
        openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();
        Index64 numQuads = 0;
        for (Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            numQuads += polygonPoolList[n].numQuads();
        }

        std::vector<GLuint> indices;
        indices.reserve(numQuads * 4);
        openvdb::Vec3d normal, e1, e2;

        for (Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
            for (Index64 i = 0, I = polygons.numQuads(); i < I; ++i) {
                const openvdb::Vec4I& quad = polygons.quad(i);
                indices.push_back(quad[0]);
                indices.push_back(quad[1]);
                indices.push_back(quad[2]);
                indices.push_back(quad[3]);

                e1 = mesher.pointList()[quad[1]];
                e1 -= mesher.pointList()[quad[0]];
                e2 = mesher.pointList()[quad[2]];
                e2 -= mesher.pointList()[quad[1]];
                normal = e1.cross(e2);

                const double length = normal.length();
                if (length > 1.0e-7) normal *= (1.0 / length);

                for (int v = 0; v < 4; ++v) {
                    normals[quad[v]*3]    = static_cast<GLfloat>(-normal[0]);
                    normals[quad[v]*3+1]  = static_cast<GLfloat>(-normal[1]);
                    normals[quad[v]*3+2]  = static_cast<GLfloat>(-normal[2]);
                }
            }
        }

        // Construct and transfer GPU buffers.
        mBuffer->genVertexBuffer(points);
        mBuffer->genNormalBuffer(normals);
        mBuffer->genIndexBuffer(indices, GL_QUADS);
    }

private:
    BufferObject *mBuffer;

    static openvdb::Vec3s sNodeColors[];

}; // MeshOp


////////////////////////////////////////

// Meshing module

MeshModule::MeshModule(const openvdb::GridBase::ConstPtr& grid):
    mGrid(grid),
    mIsInitialized(false)
{
    mShader.setVertShader(
        "#version 120\n"
        "varying vec3 normal;\n"
        "void main() {\n"
            "normal = normalize(gl_NormalMatrix * gl_Normal);\n"
            "gl_Position =  ftransform();\n"
            "gl_ClipVertex = gl_ModelViewMatrix * gl_Vertex;\n"
        "}\n");

    mShader.setFragShader(
        "#version 120\n"
        "varying vec3 normal;\n"
        "const vec4 skyColor = vec4(0.9, 0.9, 1.0, 1.0);\n"
        "const vec4 groundColor = vec4(0.3, 0.3, 0.2, 1.0);\n"
        "void main() {\n"
            "vec3 normalized_normal = normalize(normal);\n"
            "float w = 0.5 * (1.0 + dot(normalized_normal, vec3(0.0, 1.0, 0.0)));\n"
            "vec4 diffuseColor = w * skyColor + (1.0 - w) * groundColor;\n"
            "gl_FragColor = diffuseColor;\n"
        "}\n");

    mShader.build();
}


void
MeshModule::init()
{
    mIsInitialized = true;

    MeshOp drawMesh(mBufferObject);

    if (!util::processTypedScalarGrid(mGrid, drawMesh)) {
        OPENVDB_LOG_INFO(
            "Ignoring non-scalar grid type during mesh module initialization.");
    }
}


void
MeshModule::render()
{
    if (!mIsVisible) return;
    if (!mIsInitialized) init();

    mShader.startShading();

    mBufferObject.render();

    mShader.stopShading();
}

} // namespace openvdb_viewer

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
