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

#include <vector>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/tools/LevelSetUtil.h> // for sdfInteriorMask()
#include <openvdb/tools/ParticlesToLevelSet.h>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);


class TestParticlesToLevelSet: public CppUnit::TestFixture
{
public:
    virtual void setUp() {openvdb::initialize();}
    virtual void tearDown() {openvdb::uninitialize();}

    void writeGrid(openvdb::GridBase::Ptr grid, std::string fileName) const
    {
        std::cout << "\nWriting \""<<fileName<<"\" to file\n";
        grid->setName("TestParticlesToLevelSet");
        openvdb::GridPtrVec grids;
        grids.push_back(grid);
        openvdb::io::File file(fileName + ".vdb");
        file.write(grids);
        file.close();
    }

    CPPUNIT_TEST_SUITE(TestParticlesToLevelSet);
    CPPUNIT_TEST(testMyParticleList);
    CPPUNIT_TEST(testRasterizeSpheres);
    CPPUNIT_TEST(testRasterizeSpheresAndId);
    CPPUNIT_TEST(testRasterizeTrails);
    CPPUNIT_TEST(testRasterizeTrailsAndId);
    CPPUNIT_TEST(testMaskOutput);
    CPPUNIT_TEST_SUITE_END();

    void testMyParticleList();
    void testRasterizeSpheres();
    void testRasterizeSpheresAndId();
    void testRasterizeTrails();
    void testRasterizeTrailsAndId();
    void testMaskOutput();
};


CPPUNIT_TEST_SUITE_REGISTRATION(TestParticlesToLevelSet);

class MyParticleList
{
protected:
    struct MyParticle {
        openvdb::Vec3R p, v;
        openvdb::Real  r;
    };
    openvdb::Real           mRadiusScale;
    openvdb::Real           mVelocityScale;
    std::vector<MyParticle> mParticleList;
public:

    typedef openvdb::Vec3R  PosType;

    MyParticleList(openvdb::Real rScale=1, openvdb::Real vScale=1)
        : mRadiusScale(rScale), mVelocityScale(vScale) {}
    void add(const openvdb::Vec3R &p, const openvdb::Real &r,
             const openvdb::Vec3R &v=openvdb::Vec3R(0,0,0))
    {
        MyParticle pa;
        pa.p = p;
        pa.r = r;
        pa.v = v;
        mParticleList.push_back(pa);
    }
    /// @return coordinate bbox in the space of the specified transfrom
    openvdb::CoordBBox getBBox(const openvdb::GridBase& grid) {
        openvdb::CoordBBox bbox;
        openvdb::Coord &min= bbox.min(), &max = bbox.max();
        openvdb::Vec3R pos;
        openvdb::Real rad, invDx = 1/grid.voxelSize()[0];
        for (size_t n=0, e=this->size(); n<e; ++n) {
            this->getPosRad(n, pos, rad);
            const openvdb::Vec3d xyz = grid.worldToIndex(pos);
            const openvdb::Real   r  = rad * invDx;
            for (int i=0; i<3; ++i) {
                min[i] = openvdb::math::Min(min[i], openvdb::math::Floor(xyz[i] - r));
                max[i] = openvdb::math::Max(max[i], openvdb::math::Ceil( xyz[i] + r));
            }
        }
        return bbox;
    }
    //typedef int AttributeType;
    // The methods below are only required for the unit-tests
    openvdb::Vec3R pos(int n)   const {return mParticleList[n].p;}
    openvdb::Vec3R vel(int n)   const {return mVelocityScale*mParticleList[n].v;}
    openvdb::Real radius(int n) const {return mRadiusScale*mParticleList[n].r;}

    //////////////////////////////////////////////////////////////////////////////
    /// The methods below are the only ones required by tools::ParticleToLevelSet
    /// @note We return by value since the radius and velocities are modified
    /// by the scaling factors! Also these methods are all assumed to
    /// be thread-safe.

    /// Return the total number of particles in list.
    ///  Always required!
    size_t size() const { return mParticleList.size(); }

    /// Get the world space position of n'th particle.
    /// Required by ParticledToLevelSet::rasterizeSphere(*this,radius).
    void getPos(size_t n,  openvdb::Vec3R&pos) const { pos = mParticleList[n].p; }


    void getPosRad(size_t n,  openvdb::Vec3R& pos, openvdb::Real& rad) const {
        pos = mParticleList[n].p;
        rad = mRadiusScale*mParticleList[n].r;
    }
    void getPosRadVel(size_t n,  openvdb::Vec3R& pos, openvdb::Real& rad, openvdb::Vec3R& vel) const {
        pos = mParticleList[n].p;
        rad = mRadiusScale*mParticleList[n].r;
        vel = mVelocityScale*mParticleList[n].v;
    }
    // The method below is only required for attribute transfer
    void getAtt(size_t n, openvdb::Index32& att) const { att = openvdb::Index32(n); }
};


void
TestParticlesToLevelSet::testMyParticleList()
{
    MyParticleList pa;
    CPPUNIT_ASSERT_EQUAL(0, int(pa.size()));
    pa.add(openvdb::Vec3R(10,10,10), 2, openvdb::Vec3R(1,0,0));
    CPPUNIT_ASSERT_EQUAL(1, int(pa.size()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(10, pa.pos(0)[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(10, pa.pos(0)[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(10, pa.pos(0)[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(1 , pa.vel(0)[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(0)[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(0)[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(2 , pa.radius(0));
    pa.add(openvdb::Vec3R(20,20,20), 3);
    CPPUNIT_ASSERT_EQUAL(2, int(pa.size()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(20, pa.pos(1)[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(20, pa.pos(1)[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(20, pa.pos(1)[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(1)[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(1)[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(1)[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(3 , pa.radius(1));

    const float voxelSize = 0.5f, halfWidth = 4.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
    openvdb::CoordBBox bbox = pa.getBBox(*ls);
    ASSERT_DOUBLES_EXACTLY_EQUAL((10-2)/voxelSize, bbox.min()[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((10-2)/voxelSize, bbox.min()[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((10-2)/voxelSize, bbox.min()[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((20+3)/voxelSize, bbox.max()[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((20+3)/voxelSize, bbox.max()[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((20+3)/voxelSize, bbox.max()[2]);
}


void
TestParticlesToLevelSet::testRasterizeSpheres()
{
    MyParticleList pa;
    pa.add(openvdb::Vec3R(10,10,10), 2);
    pa.add(openvdb::Vec3R(20,20,20), 2);
    // testing CSG
    pa.add(openvdb::Vec3R(31.0,31,31), 5);
    pa.add(openvdb::Vec3R(31.5,31,31), 5);
    pa.add(openvdb::Vec3R(32.0,31,31), 5);
    pa.add(openvdb::Vec3R(32.5,31,31), 5);
    pa.add(openvdb::Vec3R(33.0,31,31), 5);
    pa.add(openvdb::Vec3R(33.5,31,31), 5);
    pa.add(openvdb::Vec3R(34.0,31,31), 5);
    pa.add(openvdb::Vec3R(34.5,31,31), 5);
    pa.add(openvdb::Vec3R(35.0,31,31), 5);
    pa.add(openvdb::Vec3R(35.5,31,31), 5);
    pa.add(openvdb::Vec3R(36.0,31,31), 5);
    CPPUNIT_ASSERT_EQUAL(13, int(pa.size()));

    const float voxelSize = 1.0f, halfWidth = 2.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> raster(*ls);

    raster.setGrainSize(1);//a value of zero disables threading
    raster.rasterizeSpheres(pa);
    raster.finalize();
    //openvdb::FloatGrid::Ptr ls = raster.getSdfGrid();

    //ls->tree().print(std::cout,4);
    //this->writeGrid(ls, "testRasterizeSpheres");

    ASSERT_DOUBLES_EXACTLY_EQUAL(halfWidth * voxelSize,
        ls->tree().getValue(openvdb::Coord( 0, 0, 0)));

    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord( 6,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord( 7,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord( 8,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord( 9,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-2, ls->tree().getValue(openvdb::Coord(10,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(11,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(12,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(13,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(14,10,10)));

    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(20,16,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(20,17,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(20,18,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(20,19,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-2, ls->tree().getValue(openvdb::Coord(20,20,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(20,21,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(20,22,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(20,23,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(20,24,20)));
    {// full but slow test of all voxels
        openvdb::CoordBBox bbox = pa.getBBox(*ls);
        bbox.expand(static_cast<int>(halfWidth)+1);
        openvdb::Index64 count=0;
        const float outside = ls->background(), inside = -outside;
        const openvdb::Coord &min=bbox.min(), &max=bbox.max();
        for (openvdb::Coord ijk=min; ijk[0]<max[0]; ++ijk[0]) {
            for (ijk[1]=min[1]; ijk[1]<max[1]; ++ijk[1]) {
                for (ijk[2]=min[2]; ijk[2]<max[2]; ++ijk[2]) {
                    const openvdb::Vec3d xyz = ls->indexToWorld(ijk.asVec3d());
                    double dist = (xyz-pa.pos(0)).length()-pa.radius(0);
                    for (int i = 1, s = int(pa.size()); i < s; ++i) {
                        dist=openvdb::math::Min(dist,(xyz-pa.pos(i)).length()-pa.radius(i));
                    }
                    const float val = ls->tree().getValue(ijk);
                    if (dist >= outside) {
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(outside, val, 0.0001);
                        CPPUNIT_ASSERT(ls->tree().isValueOff(ijk));
                    } else if( dist <= inside ) {
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(inside, val, 0.0001);
                        CPPUNIT_ASSERT(ls->tree().isValueOff(ijk));
                    } else {
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(  dist, val, 0.0001);
                        CPPUNIT_ASSERT(ls->tree().isValueOn(ijk));
                        ++count;
                    }
                }
            }
        }
        //std::cerr << "\nExpected active voxel count = " << count
        //    << ", actual active voxle count = "
        //    << ls->activeVoxelCount() << std::endl;
        CPPUNIT_ASSERT_EQUAL(count, ls->activeVoxelCount());
    }
}


void
TestParticlesToLevelSet::testRasterizeSpheresAndId()
{
    MyParticleList pa(0.5f);
    pa.add(openvdb::Vec3R(10,10,10), 4);
    pa.add(openvdb::Vec3R(20,20,20), 4);
    // testing CSG
    pa.add(openvdb::Vec3R(31.0,31,31),10);
    pa.add(openvdb::Vec3R(31.5,31,31),10);
    pa.add(openvdb::Vec3R(32.0,31,31),10);
    pa.add(openvdb::Vec3R(32.5,31,31),10);
    pa.add(openvdb::Vec3R(33.0,31,31),10);
    pa.add(openvdb::Vec3R(33.5,31,31),10);
    pa.add(openvdb::Vec3R(34.0,31,31),10);
    pa.add(openvdb::Vec3R(34.5,31,31),10);
    pa.add(openvdb::Vec3R(35.0,31,31),10);
    pa.add(openvdb::Vec3R(35.5,31,31),10);
    pa.add(openvdb::Vec3R(36.0,31,31),10);
    CPPUNIT_ASSERT_EQUAL(13, int(pa.size()));

    typedef openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32> RasterT;
    const float voxelSize = 1.0f, halfWidth = 2.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);

    RasterT raster(*ls);
    raster.setGrainSize(1);//a value of zero disables threading
    raster.rasterizeSpheres(pa);
    raster.finalize();
    const RasterT::AttGridType::Ptr id = raster.attributeGrid();

    int minVal = std::numeric_limits<int>::max(), maxVal = -minVal;
    for (RasterT::AttGridType::ValueOnCIter i=id->cbeginValueOn(); i; ++i) {
        minVal = openvdb::math::Min(minVal, int(*i));
        maxVal = openvdb::math::Max(maxVal, int(*i));
    }
    CPPUNIT_ASSERT_EQUAL(0 , minVal);
    CPPUNIT_ASSERT_EQUAL(12, maxVal);

    //grid.tree().print(std::cout,4);
    //id->print(std::cout,4);
    //this->writeGrid(ls, "testRasterizeSpheres");

    ASSERT_DOUBLES_EXACTLY_EQUAL(halfWidth * voxelSize,
                                 ls->tree().getValue(openvdb::Coord( 0, 0, 0)));

    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord( 6,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord( 7,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord( 8,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord( 9,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-2, ls->tree().getValue(openvdb::Coord(10,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(11,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(12,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(13,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(14,10,10)));

    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(20,16,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(20,17,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(20,18,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(20,19,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-2, ls->tree().getValue(openvdb::Coord(20,20,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(20,21,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(20,22,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(20,23,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(20,24,20)));

    {// full but slow test of all voxels
        openvdb::CoordBBox bbox = pa.getBBox(*ls);
        bbox.expand(static_cast<int>(halfWidth)+1);
        openvdb::Index64 count = 0;
        const float outside = ls->background(), inside = -outside;
        const openvdb::Coord &min=bbox.min(), &max=bbox.max();
        for (openvdb::Coord ijk=min; ijk[0]<max[0]; ++ijk[0]) {
            for (ijk[1]=min[1]; ijk[1]<max[1]; ++ijk[1]) {
                for (ijk[2]=min[2]; ijk[2]<max[2]; ++ijk[2]) {
                    const openvdb::Vec3d xyz = ls->indexToWorld(ijk.asVec3d());
                    double dist = (xyz-pa.pos(0)).length()-pa.radius(0);
                    openvdb::Index32 k =0;
                    for (int i = 1, s = int(pa.size()); i < s; ++i) {
                        double d = (xyz-pa.pos(i)).length()-pa.radius(i);
                        if (d<dist) {
                            k = openvdb::Index32(i);
                            dist = d;
                        }
                    }//loop over particles
                    const float val = ls->tree().getValue(ijk);
                    openvdb::Index32 m = id->tree().getValue(ijk);
                    if (dist >= outside) {
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(outside, val, 0.0001);
                        CPPUNIT_ASSERT(ls->tree().isValueOff(ijk));
                        //CPPUNIT_ASSERT_EQUAL(openvdb::util::INVALID_IDX, m);
                        CPPUNIT_ASSERT(id->tree().isValueOff(ijk));
                    } else if( dist <= inside ) {
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(inside, val, 0.0001);
                        CPPUNIT_ASSERT(ls->tree().isValueOff(ijk));
                        //CPPUNIT_ASSERT_EQUAL(openvdb::util::INVALID_IDX, m);
                        CPPUNIT_ASSERT(id->tree().isValueOff(ijk));
                    } else {
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(  dist, val, 0.0001);
                        CPPUNIT_ASSERT(ls->tree().isValueOn(ijk));
                        CPPUNIT_ASSERT_EQUAL(k, m);
                        CPPUNIT_ASSERT(id->tree().isValueOn(ijk));
                        ++count;
                    }
                }
            }
        }
        //std::cerr << "\nExpected active voxel count = " << count
        //    << ", actual active voxle count = "
        //    << ls->activeVoxelCount() << std::endl;
        CPPUNIT_ASSERT_EQUAL(count, ls->activeVoxelCount());
    }
}


/// This is not really a conventional unit-test since the result of
/// the tests are written to a file and need to be visually verified!
void
TestParticlesToLevelSet::testRasterizeTrails()
{
    const float voxelSize = 1.0f, halfWidth = 2.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);

    MyParticleList pa(1,5);

    // This particle radius = 1 < 1.5 i.e. it's below the Nyquist frequency and hence ignored
    pa.add(openvdb::Vec3R(  0,  0,  0), 1, openvdb::Vec3R( 0, 1, 0));
    pa.add(openvdb::Vec3R(-10,-10,-10), 2, openvdb::Vec3R( 2, 0, 0));
    pa.add(openvdb::Vec3R( 10, 10, 10), 3, openvdb::Vec3R( 0, 1, 0));
    pa.add(openvdb::Vec3R(  0,  0,  0), 6, openvdb::Vec3R( 0, 0,-5));
    pa.add(openvdb::Vec3R( 20,  0,  0), 2, openvdb::Vec3R( 0, 0, 0));

    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> raster(*ls);
    raster.rasterizeTrails(pa, 0.75);//scale offset between two instances

    //ls->tree().print(std::cout, 4);
    //this->writeGrid(ls, "testRasterizeTrails");
}


void
TestParticlesToLevelSet::testRasterizeTrailsAndId()
{
    MyParticleList pa(1,5);

    // This particle radius = 1 < 1.5 i.e. it's below the Nyquist frequency and hence ignored
    pa.add(openvdb::Vec3R(  0,  0,  0), 1, openvdb::Vec3R( 0, 1, 0));
    pa.add(openvdb::Vec3R(-10,-10,-10), 2, openvdb::Vec3R( 2, 0, 0));
    pa.add(openvdb::Vec3R( 10, 10, 10), 3, openvdb::Vec3R( 0, 1, 0));
    pa.add(openvdb::Vec3R(  0,  0,  0), 6, openvdb::Vec3R( 0, 0,-5));

    typedef openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index> RasterT;
    const float voxelSize = 1.0f, halfWidth = 2.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
    RasterT raster(*ls);
    raster.rasterizeTrails(pa, 0.75);//scale offset between two instances
    raster.finalize();
    const RasterT::AttGridType::Ptr id = raster.attributeGrid();
    CPPUNIT_ASSERT(!ls->empty());
    CPPUNIT_ASSERT(!id->empty());
    CPPUNIT_ASSERT_EQUAL(ls->activeVoxelCount(),id->activeVoxelCount());

    int min = std::numeric_limits<int>::max(), max = -min;
    for (RasterT::AttGridType::ValueOnCIter i=id->cbeginValueOn(); i; ++i) {
        min = openvdb::math::Min(min, int(*i));
        max = openvdb::math::Max(max, int(*i));
    }
    CPPUNIT_ASSERT_EQUAL(1, min);//first particle is ignored because of its small rdadius!
    CPPUNIT_ASSERT_EQUAL(3, max);

    //ls->tree().print(std::cout, 4);
    //this->writeGrid(ls, "testRasterizeTrails");
}


void
TestParticlesToLevelSet::testMaskOutput()
{
    using namespace openvdb;

    using SdfGridType = FloatGrid;
    using MaskGridType = MaskGrid;

    MyParticleList pa;
    const Vec3R vel(10, 5, 1);
    pa.add(Vec3R(84.7252, 85.7946, 84.4266), 11.8569, vel);
    pa.add(Vec3R(47.9977, 81.2169, 47.7665), 5.45313, vel);
    pa.add(Vec3R(87.0087, 14.0351, 95.7155), 7.36483, vel);
    pa.add(Vec3R(75.8616, 53.7373, 58.202),  14.4127, vel);
    pa.add(Vec3R(14.9675, 32.4141, 13.5218), 4.33101, vel);
    pa.add(Vec3R(96.9809, 9.92804, 90.2349), 12.2613, vel);
    pa.add(Vec3R(63.4274, 3.84254, 32.5047), 12.1566, vel);
    pa.add(Vec3R(62.351,  47.4698, 41.4369), 11.637,  vel);
    pa.add(Vec3R(62.2846, 1.35716, 66.2527), 18.9914, vel);
    pa.add(Vec3R(44.1711, 1.99877, 45.1159), 1.11429, vel);

    {
        // Test variable-radius particles.

        // Rasterize into an SDF.
        auto sdf = createLevelSet<SdfGridType>();
        tools::particlesToSdf(pa, *sdf);

        // Rasterize into a boolean mask.
        auto mask = MaskGridType::create();
        tools::particlesToMask(pa, *mask);

        // Verify that the rasterized mask matches the interior of the SDF.
        mask->tree().voxelizeActiveTiles();
        auto interior = tools::sdfInteriorMask(*sdf);
        CPPUNIT_ASSERT(interior);
        interior->tree().voxelizeActiveTiles();
        CPPUNIT_ASSERT_EQUAL(interior->activeVoxelCount(), mask->activeVoxelCount());
        interior->topologyDifference(*mask);
        CPPUNIT_ASSERT_EQUAL(0, int(interior->activeVoxelCount()));
    }
    {
        // Test fixed-radius particles.

        auto sdf = createLevelSet<SdfGridType>();
        tools::particlesToSdf(pa, *sdf, /*radius=*/10.0);

        auto mask = MaskGridType::create();
        tools::particlesToMask(pa, *mask, /*radius=*/10.0);

        mask->tree().voxelizeActiveTiles();
        auto interior = tools::sdfInteriorMask(*sdf);
        CPPUNIT_ASSERT(interior);
        interior->tree().voxelizeActiveTiles();
        CPPUNIT_ASSERT_EQUAL(interior->activeVoxelCount(), mask->activeVoxelCount());
        interior->topologyDifference(*mask);
        CPPUNIT_ASSERT_EQUAL(0, int(interior->activeVoxelCount()));
    }
    {
        // Test particle trails.

        auto sdf = createLevelSet<SdfGridType>();
        tools::particleTrailsToSdf(pa, *sdf);

        auto mask = MaskGridType::create();
        tools::particleTrailsToMask(pa, *mask);

        mask->tree().voxelizeActiveTiles();
        auto interior = tools::sdfInteriorMask(*sdf);
        CPPUNIT_ASSERT(interior);
        interior->tree().voxelizeActiveTiles();
        CPPUNIT_ASSERT_EQUAL(interior->activeVoxelCount(), mask->activeVoxelCount());
        interior->topologyDifference(*mask);
        CPPUNIT_ASSERT_EQUAL(0, int(interior->activeVoxelCount()));
    }
    {
        // Test attribute transfer.

        auto sdf = createLevelSet<SdfGridType>();
        tools::ParticlesToLevelSet<SdfGridType, Index32> p2sdf(*sdf);
        p2sdf.rasterizeSpheres(pa);
        p2sdf.finalize(/*prune=*/true);
        const auto sdfAttr = p2sdf.attributeGrid();
        CPPUNIT_ASSERT(sdfAttr);

        auto mask = MaskGridType::create();
        tools::ParticlesToLevelSet<MaskGridType, Index32> p2mask(*mask);
        p2mask.rasterizeSpheres(pa);
        p2mask.finalize(/*prune=*/true);
        const auto maskAttr = p2mask.attributeGrid();
        CPPUNIT_ASSERT(maskAttr);

        mask->tree().voxelizeActiveTiles();
        auto interior = tools::sdfInteriorMask(*sdf);
        CPPUNIT_ASSERT(interior);
        interior->tree().voxelizeActiveTiles();
        CPPUNIT_ASSERT_EQUAL(interior->activeVoxelCount(), mask->activeVoxelCount());
        interior->topologyDifference(*mask);
        CPPUNIT_ASSERT_EQUAL(0, int(interior->activeVoxelCount()));

        // Verify that the mask- and SDF-generated attribute grids match.
        auto sdfAcc = sdfAttr->getConstAccessor();
        auto maskAcc = maskAttr->getConstAccessor();
        for (auto it = interior->cbeginValueOn(); it; ++it) {
            const auto& c = it.getCoord();
            CPPUNIT_ASSERT_EQUAL(sdfAcc.getValue(c), maskAcc.getValue(c));
        }
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
