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

#include <openvdb/Exceptions.h>
#include <openvdb/io/Stream.h>
#include <openvdb/Metadata.h>
#include <openvdb/math/Maps.h>
#include <openvdb/math/Transform.h>
#include <openvdb/version.h>
#include <openvdb/openvdb.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cstdio> // for remove()
#include <fstream>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(a, b) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((a), (b), /*tolerance=*/0.0);


class TestStream: public CppUnit::TestCase
{
public:
    void setUp() override;
    void tearDown() override;

    CPPUNIT_TEST_SUITE(TestStream);
    CPPUNIT_TEST(testWrite);
    CPPUNIT_TEST(testRead);
    CPPUNIT_TEST(testFileReadFromStream);
    CPPUNIT_TEST_SUITE_END();

    void testWrite();
    void testRead();
    void testFileReadFromStream();

private:
    static openvdb::GridPtrVecPtr createTestGrids(openvdb::MetaMap::Ptr&);
    static void verifyTestGrids(openvdb::GridPtrVecPtr, openvdb::MetaMap::Ptr);
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestStream);


////////////////////////////////////////


void
TestStream::setUp()
{
    openvdb::uninitialize();

    openvdb::Int32Grid::registerGrid();
    openvdb::FloatGrid::registerGrid();

    openvdb::StringMetadata::registerType();
    openvdb::Int32Metadata::registerType();
    openvdb::Int64Metadata::registerType();
    openvdb::Vec3IMetadata::registerType();

    // Register maps
    openvdb::math::MapRegistry::clear();
    openvdb::math::AffineMap::registerMap();
    openvdb::math::ScaleMap::registerMap();
    openvdb::math::UniformScaleMap::registerMap();
    openvdb::math::TranslationMap::registerMap();
    openvdb::math::ScaleTranslateMap::registerMap();
    openvdb::math::UniformScaleTranslateMap::registerMap();
    openvdb::math::NonlinearFrustumMap::registerMap();
}


void
TestStream::tearDown()
{
    openvdb::uninitialize();
}


////////////////////////////////////////


openvdb::GridPtrVecPtr
TestStream::createTestGrids(openvdb::MetaMap::Ptr& metadata)
{
    using namespace openvdb;

    // Create trees
    Int32Tree::Ptr tree1(new Int32Tree(1));
    FloatTree::Ptr tree2(new FloatTree(2.0));

    // Set some values
    tree1->setValue(Coord(0, 0, 0), 5);
    tree1->setValue(Coord(100, 0, 0), 6);
    tree2->setValue(Coord(0, 0, 0), 10);
    tree2->setValue(Coord(0, 100, 0), 11);

    // Create grids
    GridBase::Ptr
        grid1 = createGrid(tree1),
        grid2 = createGrid(tree1), // instance of grid1
        grid3 = createGrid(tree2);
    grid1->setName("density");
    grid2->setName("density_copy");
    grid3->setName("temperature");

    // Create transforms
    math::Transform::Ptr trans1 = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.1);
    grid1->setTransform(trans1);
    grid2->setTransform(trans2);
    grid3->setTransform(trans2);

    metadata.reset(new MetaMap);
    metadata->insertMeta("author", StringMetadata("Einstein"));
    metadata->insertMeta("year", Int32Metadata(2009));

    GridPtrVecPtr grids(new GridPtrVec);
    grids->push_back(grid1);
    grids->push_back(grid2);
    grids->push_back(grid3);

    return grids;
}


void
TestStream::verifyTestGrids(openvdb::GridPtrVecPtr grids, openvdb::MetaMap::Ptr meta)
{
    using namespace openvdb;

    CPPUNIT_ASSERT(grids.get() != nullptr);
    CPPUNIT_ASSERT(meta.get() != nullptr);

    // Verify the metadata.
    CPPUNIT_ASSERT_EQUAL(2, int(meta->metaCount()));
    CPPUNIT_ASSERT_EQUAL(std::string("Einstein"), meta->metaValue<std::string>("author"));
    CPPUNIT_ASSERT_EQUAL(2009, meta->metaValue<int32_t>("year"));

    // Verify the grids.
    CPPUNIT_ASSERT_EQUAL(3, int(grids->size()));

    GridBase::Ptr grid = findGridByName(*grids, "density");
    CPPUNIT_ASSERT(grid.get() != nullptr);
    Int32Tree::Ptr density = gridPtrCast<Int32Grid>(grid)->treePtr();
    CPPUNIT_ASSERT(density.get() != nullptr);

    grid.reset();
    grid = findGridByName(*grids, "density_copy");
    CPPUNIT_ASSERT(grid.get() != nullptr);
    CPPUNIT_ASSERT(gridPtrCast<Int32Grid>(grid)->treePtr().get() != nullptr);
    // Verify that "density_copy" is an instance of (i.e., shares a tree with) "density".
    CPPUNIT_ASSERT_EQUAL(density, gridPtrCast<Int32Grid>(grid)->treePtr());

    grid.reset();
    grid = findGridByName(*grids, "temperature");
    CPPUNIT_ASSERT(grid.get() != nullptr);
    FloatTree::Ptr temperature = gridPtrCast<FloatGrid>(grid)->treePtr();
    CPPUNIT_ASSERT(temperature.get() != nullptr);

    ASSERT_DOUBLES_EXACTLY_EQUAL(5, density->getValue(Coord(0, 0, 0)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(6, density->getValue(Coord(100, 0, 0)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(10, temperature->getValue(Coord(0, 0, 0)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(11, temperature->getValue(Coord(0, 100, 0)));
}


////////////////////////////////////////


void
TestStream::testWrite()
{
    using namespace openvdb;

    // Create test grids and stream them to a string.
    MetaMap::Ptr meta;
    GridPtrVecPtr grids = createTestGrids(meta);
    std::ostringstream ostr(std::ios_base::binary);
    io::Stream(ostr).write(*grids, *meta);
    //std::ofstream file("debug.vdb2", std::ios_base::binary);
    //file << ostr.str();

    // Stream the grids back in.
    std::istringstream is(ostr.str(), std::ios_base::binary);
    io::Stream strm(is);
    meta = strm.getMetadata();
    grids = strm.getGrids();

    verifyTestGrids(grids, meta);
}


void
TestStream::testRead()
{
    using namespace openvdb;

    // Create test grids and write them to a file.
    MetaMap::Ptr meta;
    GridPtrVecPtr grids = createTestGrids(meta);
    const char* filename = "something.vdb2";
    io::File(filename).write(*grids, *meta);
    SharedPtr<const char> scopedFile(filename, ::remove);

    // Stream the grids back in.
    std::ifstream is(filename, std::ios_base::binary);
    io::Stream strm(is);
    meta = strm.getMetadata();
    grids = strm.getGrids();

    verifyTestGrids(grids, meta);
}


/// Stream grids to a file using io::Stream, then read the file back using io::File.
void
TestStream::testFileReadFromStream()
{
    using namespace openvdb;

    MetaMap::Ptr meta;
    GridPtrVecPtr grids;

    // Create test grids and stream them to a file (and then close the file).
    const char* filename = "something.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);
    {
        std::ofstream os(filename, std::ios_base::binary);
        grids = createTestGrids(meta);
        io::Stream(os).write(*grids, *meta);
    }

    // Read the grids back in.
    io::File file(filename);
    CPPUNIT_ASSERT(file.inputHasGridOffsets());
    CPPUNIT_ASSERT_THROW(file.getGrids(), IoError);

    file.open();
    meta = file.getMetadata();
    grids = file.getGrids();

    CPPUNIT_ASSERT(!file.inputHasGridOffsets());
    CPPUNIT_ASSERT(meta.get() != nullptr);
    CPPUNIT_ASSERT(grids.get() != nullptr);
    CPPUNIT_ASSERT(!grids->empty());

    verifyTestGrids(grids, meta);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
