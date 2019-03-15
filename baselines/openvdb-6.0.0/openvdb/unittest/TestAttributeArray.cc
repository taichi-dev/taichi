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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/AttributeSet.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/io/File.h>

#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-macros"
#endif
// Boost.Interprocess uses a header-only portion of Boost.DateTime
#define BOOST_DATE_TIME_NO_LIB
#ifdef __clang__
#pragma GCC diagnostic pop
#endif
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <tbb/tick_count.h>
#include <tbb/atomic.h>

#include <fstream>
#include <sstream>
#include <iostream>

#ifdef _MSC_VER
#include <boost/interprocess/detail/os_file_functions.hpp> // open_existing_file(), close_file()
// boost::interprocess::detail was renamed to boost::interprocess::ipcdetail in Boost 1.48.
// Ensure that both namespaces exist.
namespace boost { namespace interprocess { namespace detail {} namespace ipcdetail {} } }
#include <windows.h>
#else
#include <sys/types.h> // for struct stat
#include <sys/stat.h> // for stat()
#endif

/// @brief io::MappedFile has a private constructor, so this unit tests uses a matching proxy
class ProxyMappedFile
{
public:
    explicit ProxyMappedFile(const std::string& filename)
        : mImpl(new Impl(filename)) { }

private:
    class Impl
    {
    public:
        Impl(const std::string& filename)
            : mMap(filename.c_str(), boost::interprocess::read_only)
            , mRegion(mMap, boost::interprocess::read_only)
        {
            mLastWriteTime = 0;
            const char* regionFilename = mMap.get_name();
#ifdef _MSC_VER
            using namespace boost::interprocess::detail;
            using namespace boost::interprocess::ipcdetail;
            using openvdb::Index64;

            if (void* fh = open_existing_file(regionFilename, boost::interprocess::read_only)) {
                FILETIME mtime;
                if (GetFileTime(fh, nullptr, nullptr, &mtime)) {
                    mLastWriteTime = (Index64(mtime.dwHighDateTime) << 32) | mtime.dwLowDateTime;
                }
                close_file(fh);
            }
#else
            struct stat info;
            if (0 == ::stat(regionFilename, &info)) {
                mLastWriteTime = openvdb::Index64(info.st_mtime);
            }
#endif
        }

        using Notifier = std::function<void(std::string /*filename*/)>;
        boost::interprocess::file_mapping mMap;
        boost::interprocess::mapped_region mRegion;
        bool mAutoDelete = false;
        Notifier mNotifier;
        mutable tbb::atomic<openvdb::Index64> mLastWriteTime;
    }; // class Impl
    std::unique_ptr<Impl> mImpl;
}; // class ProxyMappedFile


/// @brief Functionality similar to openvdb::util::CpuTimer except with prefix padding and no decimals.
///
/// @code
///    ProfileTimer timer("algorithm 1");
///    // code to be timed goes here
///    timer.stop();
/// @endcode
class ProfileTimer
{
public:
    /// @brief Prints message and starts timer.
    ///
    /// @note Should normally be followed by a call to stop()
    ProfileTimer(const std::string& msg)
    {
        (void)msg;
#ifdef PROFILE
        // padd string to 50 characters
        std::string newMsg(msg);
        if (newMsg.size() < 50)     newMsg.insert(newMsg.end(), 50 - newMsg.size(), ' ');
        std::cerr << newMsg << " ... ";
#endif
        mT0 = tbb::tick_count::now();
    }

    ~ProfileTimer() { this->stop(); }

    /// Return Time diference in milliseconds since construction or start was called.
    inline double delta() const
    {
        tbb::tick_count::interval_t dt = tbb::tick_count::now() - mT0;
        return 1000.0*dt.seconds();
    }

    /// @brief Print time in milliseconds since construction or start was called.
    inline void stop() const
    {
#ifdef PROFILE
        std::stringstream ss;
        ss << std::setw(6) << ::round(this->delta());
        std::cerr << "completed in " << ss.str() << " ms\n";
#endif
    }

private:
    tbb::tick_count mT0;
};// ProfileTimer


using namespace openvdb;
using namespace openvdb::points;

class TestAttributeArray: public CppUnit::TestCase
{
public:
    void setUp() override { AttributeArray::clearRegistry(); }
    void tearDown() override { AttributeArray::clearRegistry(); }

    CPPUNIT_TEST_SUITE(TestAttributeArray);
    CPPUNIT_TEST(testFixedPointConversion);
    CPPUNIT_TEST(testRegistry);
    CPPUNIT_TEST(testAttributeArray);
    CPPUNIT_TEST(testAttributeArrayCopy);
    CPPUNIT_TEST(testAccessorEval);
    CPPUNIT_TEST(testAttributeHandle);
    CPPUNIT_TEST(testStrided);
    CPPUNIT_TEST(testDelayedLoad);
    CPPUNIT_TEST(testQuaternions);
    CPPUNIT_TEST(testMatrices);
    CPPUNIT_TEST(testProfile);

    CPPUNIT_TEST_SUITE_END();

    void testFixedPointConversion();
    void testRegistry();
    void testAttributeArray();
    void testAttributeArrayCopy();
    void testAccessorEval();
    void testAttributeHandle();
    void testStrided();
    void testDelayedLoad();
    void testQuaternions();
    void testMatrices();
    void testProfile();
}; // class TestAttributeArray

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeArray);


////////////////////////////////////////


namespace {

bool
matchingNamePairs(const openvdb::NamePair& lhs,
                  const openvdb::NamePair& rhs)
{
    if (lhs.first != rhs.first)     return false;
    if (lhs.second != rhs.second)     return false;

    return true;
}

} // namespace


////////////////////////////////////////


void
TestAttributeArray::testFixedPointConversion()
{
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(/*voxelSize=*/0.1));

    const float value = 33.5688040469035f;

    {
        // convert to fixed-point value

        const openvdb::Vec3f worldSpaceValue(value);
        const openvdb::Vec3f indexSpaceValue = transform->worldToIndex(worldSpaceValue);
        const float voxelSpaceValue = indexSpaceValue.x() - math::Round(indexSpaceValue.x()) + 0.5f;
        const uint32_t intValue = floatingPointToFixedPoint<uint32_t>(voxelSpaceValue);

        // convert back to floating-point value

        const float newVoxelSpaceValue = fixedPointToFloatingPoint<float>(intValue);
        const openvdb::Vec3f newIndexSpaceValue(newVoxelSpaceValue + math::Round(indexSpaceValue.x()) - 0.5f);
        const openvdb::Vec3f newWorldSpaceValue = transform->indexToWorld(newIndexSpaceValue);

        const float newValue = newWorldSpaceValue.x();

        CPPUNIT_ASSERT_DOUBLES_EQUAL(value, newValue, /*tolerance=*/1e-6);
    }

    {
        // convert to fixed-point value (vector)

        const openvdb::Vec3f worldSpaceValue(value, value+1, value+2);
        const openvdb::Vec3f indexSpaceValue = transform->worldToIndex(worldSpaceValue);
        const float voxelSpaceValueX = indexSpaceValue.x() - math::Round(indexSpaceValue.x()) + 0.5f;
        const float voxelSpaceValueY = indexSpaceValue.y() - math::Round(indexSpaceValue.y()) + 0.5f;
        const float voxelSpaceValueZ = indexSpaceValue.z() - math::Round(indexSpaceValue.z()) + 0.5f;
        const openvdb::Vec3f voxelSpaceValue(voxelSpaceValueX, voxelSpaceValueY, voxelSpaceValueZ);
        const openvdb::math::Vec3<uint32_t> intValue = floatingPointToFixedPoint<openvdb::math::Vec3<uint32_t>>(voxelSpaceValue);

        // convert back to floating-point value (vector)

        const openvdb::Vec3f newVoxelSpaceValue = fixedPointToFloatingPoint<openvdb::Vec3f>(intValue);
        const float newIndexSpaceValueX = newVoxelSpaceValue.x() + math::Round(indexSpaceValue.x()) - 0.5f;
        const float newIndexSpaceValueY = newVoxelSpaceValue.y() + math::Round(indexSpaceValue.y()) - 0.5f;
        const float newIndexSpaceValueZ = newVoxelSpaceValue.z() + math::Round(indexSpaceValue.z()) - 0.5f;
        const openvdb::Vec3f newIndexSpaceValue(newIndexSpaceValueX, newIndexSpaceValueY, newIndexSpaceValueZ);
        const openvdb::Vec3f newWorldSpaceValue = transform->indexToWorld(newIndexSpaceValue);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(worldSpaceValue.x(), newWorldSpaceValue.x(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(worldSpaceValue.y(), newWorldSpaceValue.y(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(worldSpaceValue.z(), newWorldSpaceValue.z(), /*tolerance=*/1e-6);

    }
}

namespace
{
// use a dummy factory as TypedAttributeArray::factory is private
static AttributeArray::Ptr factoryInt(Index n, Index strideOrTotalSize, bool constantStride)
{
    return TypedAttributeArray<int>::create(n, strideOrTotalSize, constantStride);
}
} // namespace

void
TestAttributeArray::testRegistry()
{
    using AttributeF = TypedAttributeArray<float>;
    using AttributeFTrnc = TypedAttributeArray<float, TruncateCodec>;

    AttributeArray::clearRegistry();

    { // cannot create AttributeArray that is not registered
        CPPUNIT_ASSERT(!AttributeArray::isRegistered(AttributeF::attributeType()));
        CPPUNIT_ASSERT_THROW(AttributeArray::create(AttributeF::attributeType(), Index(5)), LookupError);
    }

    { // throw when attempting to register a float type with an integer factory
        CPPUNIT_ASSERT_THROW(AttributeArray::registerType(
            AttributeF::attributeType(), factoryInt), KeyError);
    }

    // register the attribute array

    AttributeF::registerType();

    { // can register an AttributeArray with the same value type but different codec
        CPPUNIT_ASSERT_NO_THROW(AttributeFTrnc::registerType());
        CPPUNIT_ASSERT(AttributeArray::isRegistered(AttributeF::attributeType()));
        CPPUNIT_ASSERT(AttributeArray::isRegistered(AttributeFTrnc::attributeType()));
    }

    { // un-registering
        AttributeArray::unregisterType(AttributeF::attributeType());
        CPPUNIT_ASSERT(!AttributeArray::isRegistered(AttributeF::attributeType()));
        CPPUNIT_ASSERT(AttributeArray::isRegistered(AttributeFTrnc::attributeType()));
    }

    { // clearing registry
        AttributeF::registerType();
        AttributeArray::clearRegistry();
        CPPUNIT_ASSERT(!AttributeArray::isRegistered(AttributeF::attributeType()));
    }
}

void
TestAttributeArray::testAttributeArray()
{
    using AttributeArrayF = TypedAttributeArray<float>;
    using AttributeArrayD = TypedAttributeArray<double>;

    {
        AttributeArray::Ptr attr(new AttributeArrayD(50));

        CPPUNIT_ASSERT_EQUAL(Index(50), attr->size());
    }

    {
        AttributeArray::Ptr attr(new AttributeArrayD(50));

        CPPUNIT_ASSERT_EQUAL(Index(50), attr->size());

        AttributeArrayD& typedAttr = static_cast<AttributeArrayD&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), value, /*tolerance=*/double(0.0));

        // test unsafe methods for get() and set()

        typedAttr.setUnsafe(0, 1.5);
        typedAttr.getUnsafe(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.5), value, /*tolerance=*/double(0.0));

        // out-of-range get() and set()
        CPPUNIT_ASSERT_THROW(typedAttr.set(100, 0.5), openvdb::IndexError);
        CPPUNIT_ASSERT_THROW(typedAttr.set(100, 1), openvdb::IndexError);
        CPPUNIT_ASSERT_THROW(typedAttr.get(100, value), openvdb::IndexError);
        CPPUNIT_ASSERT_THROW(typedAttr.get(100), openvdb::IndexError);
    }

#ifdef NDEBUG
    { // test setUnsafe and getUnsafe on uniform arrays
        AttributeArrayD::Ptr attr(new AttributeArrayD(50));

        CPPUNIT_ASSERT_EQUAL(Index(50), attr->size());
        attr->collapse(5.0);
        CPPUNIT_ASSERT(attr->isUniform());

        CPPUNIT_ASSERT_DOUBLES_EQUAL(attr->getUnsafe(10), 5.0, /*tolerance=*/double(0.0));
        CPPUNIT_ASSERT(attr->isUniform());

        // this is expected behaviour because for performance reasons, array is not implicitly expanded

        attr->setUnsafe(10, 15.0);
        CPPUNIT_ASSERT(attr->isUniform());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(attr->getUnsafe(5), 15.0, /*tolerance=*/double(0.0));

        attr->expand();
        CPPUNIT_ASSERT(!attr->isUniform());
        attr->setUnsafe(10, 25.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(attr->getUnsafe(5), 15.0, /*tolerance=*/double(0.0));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(attr->getUnsafe(10), 25.0, /*tolerance=*/double(0.0));
    }
#endif

    using AttributeArrayC = TypedAttributeArray<double, FixedPointCodec<false>>;

    { // test hasValueType()
        AttributeArray::Ptr attrC(new AttributeArrayC(50));
        AttributeArray::Ptr attrD(new AttributeArrayD(50));
        AttributeArray::Ptr attrF(new AttributeArrayF(50));

        CPPUNIT_ASSERT(attrD->hasValueType<double>());
        CPPUNIT_ASSERT(attrC->hasValueType<double>());
        CPPUNIT_ASSERT(!attrF->hasValueType<double>());

        CPPUNIT_ASSERT(!attrD->hasValueType<float>());
        CPPUNIT_ASSERT(!attrC->hasValueType<float>());
        CPPUNIT_ASSERT(attrF->hasValueType<float>());
    }

    { // lots of type checking
#if OPENVDB_ABI_VERSION_NUMBER >= 6
        Index size(50);
        {
            TypedAttributeArray<bool> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("bool"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(1), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(1), attr.storageTypeSize());
            CPPUNIT_ASSERT(!attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(!attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<int16_t> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("int16"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(2), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(2), attr.storageTypeSize());
            CPPUNIT_ASSERT(!attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(!attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<int32_t> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("int32"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(4), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(4), attr.storageTypeSize());
            CPPUNIT_ASSERT(!attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(!attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<int64_t> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("int64"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(8), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(8), attr.storageTypeSize());
            CPPUNIT_ASSERT(!attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(!attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            // half is not registered by default, but for complete-ness
            TypedAttributeArray<half> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("half"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(2), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(2), attr.storageTypeSize());
            CPPUNIT_ASSERT(attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(!attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<float> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("float"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(4), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(4), attr.storageTypeSize());
            CPPUNIT_ASSERT(attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(!attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<double> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("double"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(8), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(8), attr.storageTypeSize());
            CPPUNIT_ASSERT(attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(!attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Vec3<int32_t>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("vec3i"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(12), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(12), attr.storageTypeSize());
            CPPUNIT_ASSERT(!attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(attr.valueTypeIsClass());
            CPPUNIT_ASSERT(attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Vec3<double>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("vec3d"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(24), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(24), attr.storageTypeSize());
            CPPUNIT_ASSERT(attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(attr.valueTypeIsClass());
            CPPUNIT_ASSERT(attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Mat3<float>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("mat3s"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(36), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(36), attr.storageTypeSize());
            CPPUNIT_ASSERT(attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Mat4<double>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("mat4d"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(128), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(128), attr.storageTypeSize());
            CPPUNIT_ASSERT(attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Quat<float>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("quats"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("null"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(16), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(16), attr.storageTypeSize());
            CPPUNIT_ASSERT(attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<float, TruncateCodec> typedAttr(size);
            AttributeArray& attr(typedAttr);
            CPPUNIT_ASSERT_EQUAL(Name("float"), attr.valueType());
            CPPUNIT_ASSERT_EQUAL(Name("trnc"), attr.codecType());
            CPPUNIT_ASSERT_EQUAL(Index(4), attr.valueTypeSize());
            CPPUNIT_ASSERT_EQUAL(Index(2), attr.storageTypeSize());
            CPPUNIT_ASSERT(attr.valueTypeIsFloatingPoint());
            CPPUNIT_ASSERT(!attr.valueTypeIsClass());
            CPPUNIT_ASSERT(!attr.valueTypeIsVector());
            CPPUNIT_ASSERT(!attr.valueTypeIsQuaternion());
            CPPUNIT_ASSERT(!attr.valueTypeIsMatrix());
        }
#endif
    }

    {
        AttributeArray::Ptr attr(new AttributeArrayC(50));

        AttributeArrayC& typedAttr = static_cast<AttributeArrayC&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), value, /*tolerance=*/double(0.0001));

        // test unsafe methods for get() and set()

        double value2 = 0.0;
        typedAttr.setUnsafe(0, double(0.2));
        typedAttr.getUnsafe(0, value2);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.2), value2, /*tolerance=*/double(0.0001));
    }

    using AttributeArrayI = TypedAttributeArray<int32_t>;

    { // Base class API

        AttributeArray::Ptr attr(new AttributeArrayI(50));

        CPPUNIT_ASSERT_EQUAL(Index(50), attr->size());

        CPPUNIT_ASSERT_EQUAL((sizeof(AttributeArrayI) + sizeof(int)), attr->memUsage());

        CPPUNIT_ASSERT(attr->isType<AttributeArrayI>());
        CPPUNIT_ASSERT(!attr->isType<AttributeArrayD>());

        CPPUNIT_ASSERT(*attr == *attr);
    }

    { // Typed class API

        const Index count = 50;
        const size_t uniformMemUsage = sizeof(AttributeArrayI) + sizeof(int);
        const size_t expandedMemUsage = sizeof(AttributeArrayI) + count * sizeof(int);

        AttributeArrayI attr(count);

        CPPUNIT_ASSERT_EQUAL(Index(count), attr.size());

        CPPUNIT_ASSERT_EQUAL(0, attr.get(0));
        CPPUNIT_ASSERT_EQUAL(0, attr.get(10));

        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        attr.set(0, 10);
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        AttributeArrayI attr2(count);
        attr2.set(0, 10);

        CPPUNIT_ASSERT(attr == attr2);

        attr.set(1, 5);

        CPPUNIT_ASSERT(!attr.compact());
        CPPUNIT_ASSERT(!attr.isUniform());

        CPPUNIT_ASSERT_EQUAL(10, attr.get(0));
        CPPUNIT_ASSERT_EQUAL(5, attr.get(1));
        CPPUNIT_ASSERT_EQUAL(0, attr.get(2));

        attr.collapse(5);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        CPPUNIT_ASSERT_EQUAL(5, attr.get(0));
        CPPUNIT_ASSERT_EQUAL(5, attr.get(20));
        CPPUNIT_ASSERT_EQUAL(5, attr.getUnsafe(20));

        attr.expand(/*fill=*/false);
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        attr.collapse(5);

        CPPUNIT_ASSERT(attr.isUniform());

        attr.expand();

        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(5, attr.get(i));
        }

        CPPUNIT_ASSERT(attr.compact());
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT(attr.compact());

        attr.expand();

        attr.fill(10);
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(10, attr.get(i));
        }

        attr.collapse(7);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        CPPUNIT_ASSERT_EQUAL(7, attr.get(0));
        CPPUNIT_ASSERT_EQUAL(7, attr.get(20));

        attr.fill(5);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(5, attr.get(i));
        }

        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());

        attr.setTransient(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());

        attr.setHidden(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());

        attr.setTransient(false);
        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());

        attr.setHidden(false);
        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());

        attr.setHidden(true);

        { // test copy construction
            AttributeArrayI attrB(attr);
            CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
            CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
            CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
            CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
            CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
            CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());
// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            CPPUNIT_ASSERT_EQUAL(attr.isCompressed(), attrB.isCompressed());
#pragma GCC diagnostic pop

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.getUnsafe(i));
                CPPUNIT_ASSERT_EQUAL(attr.getUnsafe(i), attrB.getUnsafe(i));
            }
        }

        // attribute array must not be uniform for compression

        attr.set(1, 7);
        attr.set(2, 8);
        attr.set(6, 100);

        // note that in-memory compression has been deprecated, verify all
        // isCompressed() calls return false

// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

        CPPUNIT_ASSERT(!attr.isCompressed());

        { // test compressed copy construction
            attr.compress();

            CPPUNIT_ASSERT(!attr.isCompressed());

            AttributeArray::Ptr attrCopy = attr.copy();
            AttributeArrayI& attrB(AttributeArrayI::cast(*attrCopy));

            CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
            CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
            CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
            CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
            CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
            CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());
            CPPUNIT_ASSERT_EQUAL(attr.isCompressed(), attrB.isCompressed());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.getUnsafe(i));
                CPPUNIT_ASSERT_EQUAL(attr.getUnsafe(i), attrB.getUnsafe(i));
            }
        }

        { // test compressed copy construction (uncompress on copy)
            attr.compress();

            CPPUNIT_ASSERT(!attr.isCompressed());

            AttributeArray::Ptr attrCopy = attr.copyUncompressed();
            AttributeArrayI& attrB(AttributeArrayI::cast(*attrCopy));

            CPPUNIT_ASSERT(!attrB.isCompressed());

            attr.decompress();

            CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
            CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
            CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
            CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
            CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
            CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());
            CPPUNIT_ASSERT_EQUAL(attr.isCompressed(), attrB.isCompressed());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.getUnsafe(i));
                CPPUNIT_ASSERT_EQUAL(attr.getUnsafe(i), attrB.getUnsafe(i));
            }
        }
    }

#pragma GCC diagnostic pop

    { // Fixed codec (position range)
        AttributeArray::Ptr attr1(new AttributeArrayC(50));

        AttributeArrayC& fixedPoint = static_cast<AttributeArrayC&>(*attr1);

        // position range is -0.5 => 0.5

        fixedPoint.set(0, -0.6);
        fixedPoint.set(1, -0.4);
        fixedPoint.set(2, 0.4);
        fixedPoint.set(3, 0.6);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-0.5), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-0.4), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.4), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), fixedPoint.get(3), /*tolerance=*/double(0.0001));
    }

    using UnitFixedPointCodec8 = FixedPointCodec<false, UnitRange>;
    using AttributeArrayUFxpt8 = TypedAttributeArray<float, UnitFixedPointCodec8>;

    { // 8-bit fixed codec (unit range)
        AttributeArray::Ptr attr1(new AttributeArrayUFxpt8(50));

        AttributeArrayUFxpt8& fixedPoint = static_cast<AttributeArrayUFxpt8&>(*attr1);

        // unit range is 0.0 => 1.0

        fixedPoint.set(0, -0.2);
        fixedPoint.set(1, 0.3);
        fixedPoint.set(2, 0.6);
        fixedPoint.set(3, 1.1);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.0), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.3), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.6), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0), fixedPoint.get(3), /*tolerance=*/double(0.0001));
    }

    using UnitFixedPointCodec16 = FixedPointCodec<false, UnitRange>;
    using AttributeArrayUFxpt16 = TypedAttributeArray<float, UnitFixedPointCodec16>;

    { // 16-bit fixed codec (unit range)
        AttributeArray::Ptr attr1(new AttributeArrayUFxpt16(50));

        AttributeArrayUFxpt16& fixedPoint = static_cast<AttributeArrayUFxpt16&>(*attr1);

        // unit range is 0.0 => 1.0

        fixedPoint.set(0, -0.2);
        fixedPoint.set(1, 0.3);
        fixedPoint.set(2, 0.6);
        fixedPoint.set(3, 1.1);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.0), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.3), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.6), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0), fixedPoint.get(3), /*tolerance=*/double(0.0001));
    }

    using AttributeArrayU = TypedAttributeArray<openvdb::Vec3f, UnitVecCodec>;

    { // UnitVec codec test
        AttributeArray::Ptr attr1(new AttributeArrayU(50));

        AttributeArrayU& unitVec = static_cast<AttributeArrayU&>(*attr1);

        // all vectors must be unit length

        const openvdb::Vec3f vec1(1.0, 0.0, 0.0);
        const openvdb::Vec3f vec2(openvdb::Vec3f(1.0, 2.0, 3.0).unit());
        const openvdb::Vec3f vec3(openvdb::Vec3f(1.0, 2.0, 300000.0).unit());

        unitVec.set(0, vec1);
        unitVec.set(1, vec2);
        unitVec.set(2, vec3);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec1.x()), unitVec.get(0).x(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec1.y()), unitVec.get(0).y(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec1.z()), unitVec.get(0).z(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec2.x()), unitVec.get(1).x(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec2.y()), unitVec.get(1).y(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec2.z()), unitVec.get(1).z(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec3.x()), unitVec.get(2).x(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec3.y()), unitVec.get(2).y(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec3.z()), unitVec.get(2).z(), /*tolerance=*/double(0.0001));
    }

    { // IO
        const Index count = 50;
        AttributeArrayI attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        attrA.setHidden(true);

        std::ostringstream ostr(std::ios_base::binary);
        io::setDataCompression(ostr, io::COMPRESS_BLOSC);

        attrA.write(ostr);

        AttributeArrayI attrB;

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        CPPUNIT_ASSERT(matchingNamePairs(attrA.type(), attrB.type()));
        CPPUNIT_ASSERT_EQUAL(attrA.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attrA.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attrA.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attrA.isHidden(), attrB.isHidden());
// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        CPPUNIT_ASSERT_EQUAL(attrA.isCompressed(), attrB.isCompressed());
#pragma GCC diagnostic pop
        CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
        }

        AttributeArrayI attrC(count, 3);
        attrC.setTransient(true);

        std::ostringstream ostrC(std::ios_base::binary);
        attrC.write(ostrC);

        CPPUNIT_ASSERT_EQUAL(size_t(0), ostrC.str().size());

        std::ostringstream ostrD(std::ios_base::binary);
        attrC.write(ostrD, /*transient=*/true);

        CPPUNIT_ASSERT(ostrD.str().size() != size_t(0));
    }

    // Registry
    AttributeArrayI::registerType();

    AttributeArray::Ptr attr =
        AttributeArray::create(
            AttributeArrayI::attributeType(), 34);

    { // Casting
        AttributeArray::Ptr array = TypedAttributeArray<float>::create(0);
        CPPUNIT_ASSERT_NO_THROW(TypedAttributeArray<float>::cast(*array));
        CPPUNIT_ASSERT_THROW(TypedAttributeArray<int>::cast(*array), TypeError);

        AttributeArray::ConstPtr constArray = array;
        CPPUNIT_ASSERT_NO_THROW(TypedAttributeArray<float>::cast(*constArray));
        CPPUNIT_ASSERT_THROW(TypedAttributeArray<int>::cast(*constArray), TypeError);
    }
}

struct VectorWrapper
{
    using T = std::vector<std::pair<Index, Index>>;

    VectorWrapper(const T& _data) : data(_data) { }
    operator bool() const { return index < data.size(); }
    VectorWrapper& operator++() { index++; return *this; }
    Index sourceIndex() const { assert(*this); return data[index].first; }
    Index targetIndex() const { assert(*this); return data[index].second; }

private:
    const T& data;
    T::size_type index = 0;
}; // struct VectorWrapper

void
TestAttributeArray::testAttributeArrayCopy()
{
    using AttributeArrayD = TypedAttributeArray<double>;

    Index size(50);

    // initialize some test data

    AttributeArrayD sourceTypedAttr(size);
    AttributeArray& sourceAttr(sourceTypedAttr);
    CPPUNIT_ASSERT_EQUAL(size, sourceAttr.size());

    sourceAttr.expand();
    for (Index i = 0; i < size; i++) {
        sourceTypedAttr.set(i, double(i)/2);
    }

    // initialize source -> target pairs that reverse the order

    std::vector<std::pair<Index, Index>> indexPairs;
    for (Index i = 0; i < size; i++) {
        indexPairs.push_back(std::make_pair(i, size-i-1));
    }

    // create a new index pair wrapper

    VectorWrapper wrapper(indexPairs);

    // build a target attribute array

    AttributeArrayD targetTypedAttr(size);
    AttributeArray& targetAttr(targetTypedAttr);
    for (const auto& pair : indexPairs) {
        targetTypedAttr.set(pair.second, sourceTypedAttr.get(pair.first));
    }

    { // verify behaviour with slow virtual function
        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

// disable deprecated warnings for virtual set() method (from ABI=6 onwards)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        for (const auto& pair : indexPairs) {
            attr.set(pair.second, sourceAttr, pair.first);
        }
#pragma GCC diagnostic pop

        CPPUNIT_ASSERT(targetAttr == attr);
    }

#if OPENVDB_ABI_VERSION_NUMBER >= 6
    using AttributeArrayF = TypedAttributeArray<float>;

    { // use std::vector<std::pair<Index, Index>>::begin() as iterator to AttributeArray::copy()
        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

        attr.copyValues(sourceAttr, wrapper);

        CPPUNIT_ASSERT(targetAttr == attr);
    }

    { // attempt to copy values between attribute arrays with different storage sizes
        AttributeArrayF typedAttr(size);
        AttributeArray& attr(typedAttr);

        CPPUNIT_ASSERT_THROW(attr.copyValues(sourceAttr, wrapper), TypeError);
    }

    { // attempt to copy values between integer and float attribute arrays
        AttributeArrayF typedAttr(size);
        AttributeArray& attr(typedAttr);

        CPPUNIT_ASSERT_THROW(attr.copyValues(sourceAttr, wrapper), TypeError);
    }

    { // copy values between attribute arrays with different value types, but the same storage type
        // target half array
        TypedAttributeArray<half> targetTypedAttr1(size);
        AttributeArray& targetAttr1(targetTypedAttr1);
        for (Index i = 0; i < size; i++) {
            targetTypedAttr1.set(i, sourceTypedAttr.get(i));
        }

        // truncated float array
        TypedAttributeArray<float, TruncateCodec> targetTypedAttr2(size);
        AttributeArray& targetAttr2(targetTypedAttr2);

        targetAttr2.copyValues(targetAttr1, wrapper);

        // equality fails as attribute types are not the same
        CPPUNIT_ASSERT(targetAttr2 != targetAttr);
        CPPUNIT_ASSERT(targetAttr2.type() != targetAttr.type());
        // however testing value equality succeeds
        for (Index i = 0; i < size; i++) {
            CPPUNIT_ASSERT(targetTypedAttr2.get(i) == targetTypedAttr.get(i));
        }
    }

    { // out-of-range checking
        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

        decltype(indexPairs) rangeIndexPairs(indexPairs);

        rangeIndexPairs[10].first = size+1;

        VectorWrapper rangeWrapper(rangeIndexPairs);

        CPPUNIT_ASSERT_THROW(attr.copyValues(sourceAttr, rangeWrapper), IndexError);

        rangeIndexPairs[10].first = 0;

        CPPUNIT_ASSERT_NO_THROW(attr.copyValues(sourceAttr, rangeWrapper));

        rangeIndexPairs[10].second = size+1;

        CPPUNIT_ASSERT_THROW(attr.copyValues(sourceAttr, rangeWrapper), IndexError);
    }

    { // source attribute array is uniform
        AttributeArrayD uniformTypedAttr(size);
        AttributeArray& uniformAttr(uniformTypedAttr);

        uniformTypedAttr.collapse(5.3);

        CPPUNIT_ASSERT(uniformAttr.isUniform());

        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

        CPPUNIT_ASSERT(attr.isUniform());

        attr.copyValues(uniformAttr, wrapper);

        CPPUNIT_ASSERT(attr.isUniform());

        attr.copyValues(uniformAttr, wrapper, /*preserveUniformity=*/false);

        CPPUNIT_ASSERT(!attr.isUniform());

        typedAttr.collapse(1.4);

        CPPUNIT_ASSERT(attr.isUniform());

        // resize the vector to be smaller than the size of the array

        decltype(indexPairs) subsetIndexPairs(indexPairs);
        subsetIndexPairs.resize(size-1);

        decltype(wrapper) subsetWrapper(subsetIndexPairs);

        // now copy the values attempting to preserve uniformity

        attr.copyValues(uniformAttr, subsetWrapper, /*preserveUniformity=*/true);

        // verify that the array cannot be kept uniform

        CPPUNIT_ASSERT(!attr.isUniform());
    }

    { // target attribute array is uniform
        AttributeArrayD uniformTypedAttr(size);
        AttributeArray& uniformAttr(uniformTypedAttr);

        uniformTypedAttr.collapse(5.3);

        CPPUNIT_ASSERT(uniformAttr.isUniform());

        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

        typedAttr.set(5, 1.2);
        typedAttr.set(10, 3.1);

        CPPUNIT_ASSERT(!attr.isUniform());

        std::vector<std::pair<Index, Index>> uniformIndexPairs;
        uniformIndexPairs.push_back(std::make_pair(10, 0));
        uniformIndexPairs.push_back(std::make_pair(5, 0));
        VectorWrapper uniformWrapper(uniformIndexPairs);

        // note that calling copyValues() will implicitly expand the uniform target

        CPPUNIT_ASSERT_NO_THROW(uniformAttr.copyValuesUnsafe(attr, uniformWrapper));

        CPPUNIT_ASSERT(uniformAttr.isUniform());
        CPPUNIT_ASSERT(uniformTypedAttr.get(0) == typedAttr.get(5));
    }
#endif
}


void
TestAttributeArray::testAccessorEval()
{
    using AttributeF = TypedAttributeArray<float>;

    struct TestAccessor
    {
        static float getterError(const AttributeArray* /*array*/, const Index /*n*/) {
            OPENVDB_THROW(NotImplementedError, "");
        }
        static void setterError [[noreturn]] (AttributeArray* /*array*/,
            const Index /*n*/, const float& /*value*/)
        {
            OPENVDB_THROW(NotImplementedError, "");
        }

        //static float testGetter(const AttributeArray* array, const Index n) {
        //    return AccessorEval<UnknownCodec, float>::get(&getterError, array, n);
        //}
        //static void testSetter(AttributeArray* array, const Index n, const float& value) {
        //    AccessorEval<UnknownCodec, float>::set(&setterError, array, n, value);
        //}
    };

    { // test get and set (NullCodec)
        AttributeF::Ptr attr = AttributeF::create(10);
        attr->collapse(5.0f);
        attr->expand();

        AttributeArray& array = *attr;

        // explicit codec is used here so getter and setter are not called

        AttributeWriteHandle<float, NullCodec> writeHandle(array);

        writeHandle.mSetter = TestAccessor::setterError;

        writeHandle.set(4, 15.0f);

        AttributeHandle<float, NullCodec> handle(array);

        const AttributeArray& constArray(array);
        CPPUNIT_ASSERT_EQUAL(&constArray, &handle.array());

        handle.mGetter = TestAccessor::getterError;

        const float result1 = handle.get(4);
        const float result2 = handle.get(6);

        CPPUNIT_ASSERT_EQUAL(15.0f, result1);
        CPPUNIT_ASSERT_EQUAL(5.0f, result2);
    }

    { // test get and set (UnknownCodec)
        AttributeF::Ptr attr = AttributeF::create(10);
        attr->collapse(5.0f);
        attr->expand();

        AttributeArray& array = *attr;

        // unknown codec is used here so getter and setter are called

        AttributeWriteHandle<float, UnknownCodec> writeHandle(array);

        CPPUNIT_ASSERT_EQUAL(&array, &writeHandle.array());

        writeHandle.mSetter = TestAccessor::setterError;

        CPPUNIT_ASSERT_THROW(writeHandle.set(4, 15.0f), NotImplementedError);

        AttributeHandle<float, UnknownCodec> handle(array);

        handle.mGetter = TestAccessor::getterError;

        CPPUNIT_ASSERT_THROW(handle.get(4), NotImplementedError);
    }
}


void
TestAttributeArray::testAttributeHandle()
{
    using namespace openvdb::math;

    using AttributeI            = TypedAttributeArray<int>;
    using AttributeFH           = TypedAttributeArray<float, TruncateCodec>;
    using AttributeVec3f        = TypedAttributeArray<Vec3f>;

    using AttributeHandleRWI    = AttributeWriteHandle<int>;

    AttributeI::registerType();
    AttributeFH::registerType();
    AttributeVec3f::registerType();

    // create a Descriptor and AttributeSet

    using Descriptor = AttributeSet::Descriptor;
    Descriptor::Ptr descr = Descriptor::create(AttributeVec3f::attributeType());

    unsigned count = 500;
    AttributeSet attrSet(descr, /*arrayLength=*/count);

    attrSet.appendAttribute("truncate", AttributeFH::attributeType());
    attrSet.appendAttribute("int", AttributeI::attributeType());

    // check uniform value implementation

    {
        AttributeArray* array = attrSet.get(2);

        AttributeHandleRWI nonExpandingHandle(*array, /*expand=*/false);
        CPPUNIT_ASSERT(nonExpandingHandle.isUniform());

        AttributeHandleRWI handle(*array);
        CPPUNIT_ASSERT(!handle.isUniform());

        CPPUNIT_ASSERT_EQUAL(array->size(), handle.size());

        CPPUNIT_ASSERT_EQUAL(0, handle.get(0));
        CPPUNIT_ASSERT_EQUAL(0, handle.get(10));

        handle.set(0, 10);
        CPPUNIT_ASSERT(!handle.isUniform());

        handle.collapse(5);
        CPPUNIT_ASSERT(handle.isUniform());

        CPPUNIT_ASSERT_EQUAL(5, handle.get(0));
        CPPUNIT_ASSERT_EQUAL(5, handle.get(20));

        handle.expand();
        CPPUNIT_ASSERT(!handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(5, handle.get(i));
        }

        CPPUNIT_ASSERT(handle.compact());
        CPPUNIT_ASSERT(handle.isUniform());

        handle.expand();

        handle.fill(10);
        CPPUNIT_ASSERT(!handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(10, handle.get(i));
        }

        handle.collapse(7);
        CPPUNIT_ASSERT(handle.isUniform());

        CPPUNIT_ASSERT_EQUAL(7, handle.get(0));
        CPPUNIT_ASSERT_EQUAL(7, handle.get(20));

        handle.fill(5);
        CPPUNIT_ASSERT(handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(5, handle.get(i));
        }

        CPPUNIT_ASSERT(handle.isUniform());
    }

    {
        AttributeArray* array = attrSet.get(0);

        AttributeWriteHandle<Vec3f> handle(*array);

        handle.set(5, Vec3f(10));

        CPPUNIT_ASSERT_EQUAL(Vec3f(10), handle.get(5));
    }

// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    {
        AttributeArray* array = attrSet.get(1);

        array->compress();

        AttributeWriteHandle<float> handle(*array);

        handle.set(6, float(11));

        CPPUNIT_ASSERT_EQUAL(float(11), handle.get(6));

        CPPUNIT_ASSERT(!array->isCompressed());

        array->compress();

        CPPUNIT_ASSERT(!array->isCompressed());

        {
            AttributeHandle<float> handleRO(*array);

            CPPUNIT_ASSERT(!array->isCompressed());

            CPPUNIT_ASSERT_EQUAL(float(11), handleRO.get(6));

            CPPUNIT_ASSERT(!array->isCompressed());
        }

        CPPUNIT_ASSERT(!array->isCompressed());

        {
            AttributeHandle<float> handleRO(*array, /*preserveCompression=*/false);

            // AttributeHandle uncompresses data on construction

            CPPUNIT_ASSERT(!array->isCompressed());

            CPPUNIT_ASSERT_EQUAL(float(11), handleRO.get(6));

            CPPUNIT_ASSERT(!array->isCompressed());
        }

        CPPUNIT_ASSERT(!array->isCompressed());
    }

#pragma GCC diagnostic pop

    // check values have been correctly set without using handles

    {
        AttributeVec3f* array = static_cast<AttributeVec3f*>(attrSet.get(0));

        CPPUNIT_ASSERT(array);

        CPPUNIT_ASSERT_EQUAL(Vec3f(10), array->get(5));
    }

    {
        AttributeFH* array = static_cast<AttributeFH*>(attrSet.get(1));

        CPPUNIT_ASSERT(array);

        CPPUNIT_ASSERT_EQUAL(float(11), array->get(6));
    }
}

void
TestAttributeArray::testStrided()
{
    using AttributeArrayI       = TypedAttributeArray<int>;
    using StridedHandle         = AttributeHandle<int, /*CodecType=*/UnknownCodec>;
    using StridedWriteHandle    = AttributeWriteHandle<int, /*CodecType=*/UnknownCodec>;

    { // non-strided array
        AttributeArrayI::Ptr array = AttributeArrayI::create(/*n=*/2, /*stride=*/1);
        CPPUNIT_ASSERT(array->hasConstantStride());
        CPPUNIT_ASSERT_EQUAL(Index(1), array->stride());
        CPPUNIT_ASSERT_EQUAL(Index(2), array->size());
        CPPUNIT_ASSERT_EQUAL(Index(2), array->dataSize());
    }

    { // strided array
        AttributeArrayI::Ptr array = AttributeArrayI::create(/*n=*/2, /*stride=*/3);

        CPPUNIT_ASSERT(array->hasConstantStride());

        CPPUNIT_ASSERT_EQUAL(Index(3), array->stride());
        CPPUNIT_ASSERT_EQUAL(Index(2), array->size());
        CPPUNIT_ASSERT_EQUAL(Index(6), array->dataSize());
        CPPUNIT_ASSERT(array->isUniform());

        CPPUNIT_ASSERT_EQUAL(0, array->get(0));
        CPPUNIT_ASSERT_EQUAL(0, array->get(5));
        CPPUNIT_ASSERT_THROW(array->get(6), IndexError); // out-of-range

        CPPUNIT_ASSERT_NO_THROW(StridedHandle::create(*array));
        CPPUNIT_ASSERT_NO_THROW(StridedWriteHandle::create(*array));

        array->collapse(10);

        CPPUNIT_ASSERT_EQUAL(int(10), array->get(0));
        CPPUNIT_ASSERT_EQUAL(int(10), array->get(5));

        array->expand();

        CPPUNIT_ASSERT_EQUAL(int(10), array->get(0));
        CPPUNIT_ASSERT_EQUAL(int(10), array->get(5));

        array->collapse(0);

        CPPUNIT_ASSERT_EQUAL(int(0), array->get(0));
        CPPUNIT_ASSERT_EQUAL(int(0), array->get(5));

        StridedWriteHandle writeHandle(*array);

        writeHandle.set(0, 2, 5);
        writeHandle.set(1, 1, 10);

        CPPUNIT_ASSERT_EQUAL(Index(3), writeHandle.stride());
        CPPUNIT_ASSERT_EQUAL(Index(2), writeHandle.size());

        // non-interleaved: 0 0 5 0 10 0

        CPPUNIT_ASSERT_EQUAL(5, array->get(2));
        CPPUNIT_ASSERT_EQUAL(10, array->get(4));

        CPPUNIT_ASSERT_EQUAL(5, writeHandle.get(0, 2));
        CPPUNIT_ASSERT_EQUAL(10, writeHandle.get(1, 1));

        StridedHandle handle(*array);
        CPPUNIT_ASSERT(handle.hasConstantStride());

        CPPUNIT_ASSERT_EQUAL(5, handle.get(0, 2));
        CPPUNIT_ASSERT_EQUAL(10, handle.get(1, 1));

        CPPUNIT_ASSERT_EQUAL(Index(3), handle.stride());
        CPPUNIT_ASSERT_EQUAL(Index(2), handle.size());

// as of ABI=6, the base memory requirements of an AttributeArray have been lowered
#if OPENVDB_ABI_VERSION_NUMBER >= 6
        size_t arrayMem = 40;
#else
        size_t arrayMem = 64;
#endif
        CPPUNIT_ASSERT_EQUAL(sizeof(int) * /*size*/3 * /*stride*/2 + arrayMem, array->memUsage());
    }

    { // dynamic stride
        AttributeArrayI::Ptr array = AttributeArrayI::create(
            /*n=*/2, /*stride=*/7, /*constantStride=*/false);

        CPPUNIT_ASSERT(!array->hasConstantStride());

        // zero indicates dynamic striding
        CPPUNIT_ASSERT_EQUAL(Index(0), array->stride());
        CPPUNIT_ASSERT_EQUAL(Index(2), array->size());
        // the actual array size
        CPPUNIT_ASSERT_EQUAL(Index(7), array->dataSize());
        CPPUNIT_ASSERT(array->isUniform());

        CPPUNIT_ASSERT_EQUAL(0, array->get(0));
        CPPUNIT_ASSERT_EQUAL(0, array->get(6));
        CPPUNIT_ASSERT_THROW(array->get(7), IndexError); // out-of-range

        CPPUNIT_ASSERT_NO_THROW(StridedHandle::create(*array));
        CPPUNIT_ASSERT_NO_THROW(StridedWriteHandle::create(*array));

        // handle is bound as if a linear array with stride 1
        StridedHandle handle(*array);
        CPPUNIT_ASSERT(!handle.hasConstantStride());
        CPPUNIT_ASSERT_EQUAL(Index(1), handle.stride());
        CPPUNIT_ASSERT_EQUAL(array->dataSize(), handle.size());
    }
}

void
TestAttributeArray::testDelayedLoad()
{
    using AttributeArrayI = TypedAttributeArray<int>;
    using AttributeArrayF = TypedAttributeArray<float>;

    AttributeArrayI::registerType();
    AttributeArrayF::registerType();

    io::StreamMetadata::Ptr streamMetadata(new io::StreamMetadata);

    std::string tempDir;
    if (const char* dir = std::getenv("TMPDIR")) tempDir = dir;
#ifdef _MSC_VER
    if (tempDir.empty()) {
        char tempDirBuffer[MAX_PATH+1];
        int tempDirLen = GetTempPath(MAX_PATH+1, tempDirBuffer);
        CPPUNIT_ASSERT(tempDirLen > 0 && tempDirLen <= MAX_PATH);
        tempDir = tempDirBuffer;
    }
#else
    if (tempDir.empty()) tempDir = P_tmpdir;
#endif

    { // IO
        const Index count = 50;
        AttributeArrayI attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        AttributeArrayF attrA2(count);

        std::string filename;

        // write out attribute array to a temp file
        {
            filename = tempDir + "/openvdb_delayed1";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrA.writeMetadata(fileout, false, /*paged=*/true);
            compression::PagedOutputStream outputStreamSize(fileout);
            outputStreamSize.setSizeOnly(true);
            attrA.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout);
            outputStream.setSizeOnly(false);
            attrA.writePagedBuffers(outputStream, false);
            outputStream.flush();

            attrA2.writeMetadata(fileout, false, /*paged=*/true);
            compression::PagedOutputStream outputStreamSize2(fileout);
            outputStreamSize2.setSizeOnly(true);
            attrA2.writePagedBuffers(outputStreamSize2, false);
            outputStreamSize2.flush();
            compression::PagedOutputStream outputStream2(fileout);
            outputStream2.setSizeOnly(false);
            attrA2.writePagedBuffers(outputStream2, false);
            outputStream2.flush();

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        ProxyMappedFile* proxy = new ProxyMappedFile(filename);
        SharedPtr<io::MappedFile> mappedFile(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check manual loading of data
        {
            AttributeArrayI attrB;
            AttributeArrayF attrB2;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(matchingNamePairs(attrA.type(), attrB.type()));
            CPPUNIT_ASSERT_EQUAL(attrA.size(), attrB.size());
            CPPUNIT_ASSERT_EQUAL(attrA.isUniform(), attrB.isUniform());
            CPPUNIT_ASSERT_EQUAL(attrA.isTransient(), attrB.isTransient());
            CPPUNIT_ASSERT_EQUAL(attrA.isHidden(), attrB.isHidden());
// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            CPPUNIT_ASSERT_EQUAL(attrA.isCompressed(), attrB.isCompressed());
#pragma GCC diagnostic pop

            AttributeArrayI attrBcopy(attrB);
            AttributeArrayI attrBequal = attrB;

            CPPUNIT_ASSERT(attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrBcopy.isOutOfCore());
            CPPUNIT_ASSERT(attrBequal.isOutOfCore());

#if OPENVDB_ABI_VERSION_NUMBER >= 6
            CPPUNIT_ASSERT(!static_cast<AttributeArray&>(attrB).isDataLoaded());
            CPPUNIT_ASSERT(!static_cast<AttributeArray&>(attrBcopy).isDataLoaded());
            CPPUNIT_ASSERT(!static_cast<AttributeArray&>(attrBequal).isDataLoaded());
#endif

            attrB.loadData();
            attrBcopy.loadData();
            attrBequal.loadData();

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrBcopy.isOutOfCore());
            CPPUNIT_ASSERT(!attrBequal.isOutOfCore());

#if OPENVDB_ABI_VERSION_NUMBER >= 6
            CPPUNIT_ASSERT(static_cast<AttributeArray&>(attrB).isDataLoaded());
            CPPUNIT_ASSERT(static_cast<AttributeArray&>(attrBcopy).isDataLoaded());
            CPPUNIT_ASSERT(static_cast<AttributeArray&>(attrBequal).isDataLoaded());
#endif

            CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());
            CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrBcopy.memUsage());
            CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrBequal.memUsage());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrBcopy.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrBequal.get(i));
            }

            attrB2.readMetadata(filein);
            compression::PagedInputStream inputStream2(filein);
            inputStream2.setSizeOnly(true);
            attrB2.readPagedBuffers(inputStream2);
            inputStream2.setSizeOnly(false);
            attrB2.readPagedBuffers(inputStream2);

            CPPUNIT_ASSERT(matchingNamePairs(attrA2.type(), attrB2.type()));
            CPPUNIT_ASSERT_EQUAL(attrA2.size(), attrB2.size());
            CPPUNIT_ASSERT_EQUAL(attrA2.isUniform(), attrB2.isUniform());
            CPPUNIT_ASSERT_EQUAL(attrA2.isTransient(), attrB2.isTransient());
            CPPUNIT_ASSERT_EQUAL(attrA2.isHidden(), attrB2.isHidden());
// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            CPPUNIT_ASSERT_EQUAL(attrA2.isCompressed(), attrB2.isCompressed());
#pragma GCC diagnostic pop

            AttributeArrayF attrB2copy(attrB2);
            AttributeArrayF attrB2equal = attrB2;

            CPPUNIT_ASSERT(attrB2.isOutOfCore());
            CPPUNIT_ASSERT(attrB2copy.isOutOfCore());
            CPPUNIT_ASSERT(attrB2equal.isOutOfCore());
            attrB2.loadData();
            attrB2copy.loadData();
            attrB2equal.loadData();

            CPPUNIT_ASSERT(!attrB2.isOutOfCore());
            CPPUNIT_ASSERT(!attrB2copy.isOutOfCore());
            CPPUNIT_ASSERT(!attrB2equal.isOutOfCore());

            CPPUNIT_ASSERT_EQUAL(attrA2.memUsage(), attrB2.memUsage());
            CPPUNIT_ASSERT_EQUAL(attrA2.memUsage(), attrB2copy.memUsage());
            CPPUNIT_ASSERT_EQUAL(attrA2.memUsage(), attrB2equal.memUsage());

            CPPUNIT_ASSERT_EQUAL(attrA2.get(0), attrB2.get(0));
            CPPUNIT_ASSERT_EQUAL(attrA2.get(0), attrB2copy.get(0));
            CPPUNIT_ASSERT_EQUAL(attrA2.get(0), attrB2equal.get(0));
        }

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            CPPUNIT_ASSERT(!attrB.isUniform());

            attrB.fill(5);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(5, attrB.get(i));
            }
        }

        // read in using delayed load and check streaming (write handle)
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            CPPUNIT_ASSERT(!attrB.isUniform());

            attrB.setStreaming(true);

            {
                AttributeWriteHandle<int> handle(attrB);
                CPPUNIT_ASSERT(!attrB.isOutOfCore());
                CPPUNIT_ASSERT(!attrB.isUniform());
            }

            CPPUNIT_ASSERT(!attrB.isUniform());
        }

        // read in using delayed load and check streaming (read handle)
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            CPPUNIT_ASSERT(!attrB.isUniform());

            attrB.setStreaming(true);

            {
                AttributeHandle<int> handle(attrB);
                CPPUNIT_ASSERT(!attrB.isOutOfCore());
                CPPUNIT_ASSERT(!attrB.isUniform());
            }

            CPPUNIT_ASSERT(attrB.isUniform());
        }

        // read in using delayed load and check implicit load through get()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            attrB.get(0);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
            }
        }

        // read in using delayed load and check implicit load through compress()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            CPPUNIT_ASSERT(!attrB.isCompressed());

            attrB.compress();

            CPPUNIT_ASSERT(attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrB.isCompressed());
#pragma GCC diagnostic pop
        }

        // read in using delayed load and check copy and assignment constructors
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            AttributeArrayI attrC(attrB);
            AttributeArrayI attrD = attrB;

            CPPUNIT_ASSERT(attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrC.isOutOfCore());
            CPPUNIT_ASSERT(attrD.isOutOfCore());

            attrB.loadData();
            attrC.loadData();
            attrD.loadData();

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrC.isOutOfCore());
            CPPUNIT_ASSERT(!attrD.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrC.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrD.get(i));
            }
        }

        // read in using delayed load and check implicit load through AttributeHandle
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            AttributeHandle<int> handle(attrB);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
        }

        // read in using delayed load and check detaching of file (using collapse())
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            CPPUNIT_ASSERT(!attrB.isUniform());

            attrB.collapse();

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrB.isUniform());

            CPPUNIT_ASSERT_EQUAL(0, attrB.get(0));
        }

        // read in and write out using delayed load to check writing out-of-core attributes
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            std::string filename2 = tempDir + "/openvdb_delayed5";
            std::ofstream fileout2(filename2.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout2, streamMetadata);
            io::setDataCompression(fileout2, io::COMPRESS_BLOSC);

            attrB.writeMetadata(fileout2, false, /*paged=*/true);
            compression::PagedOutputStream outputStreamSize(fileout2);
            outputStreamSize.setSizeOnly(true);
            attrB.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout2);
            outputStream.setSizeOnly(false);
            attrB.writePagedBuffers(outputStream, false);
            outputStream.flush();

            fileout2.close();

            AttributeArrayI attrB2;

            std::ifstream filein2(filename2.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein2, streamMetadata);
            io::setMappedFilePtr(filein2, mappedFile);

            attrB2.readMetadata(filein2);
            compression::PagedInputStream inputStream2(filein2);
            inputStream2.setSizeOnly(true);
            attrB2.readPagedBuffers(inputStream2);
            inputStream2.setSizeOnly(false);
            attrB2.readPagedBuffers(inputStream2);

            CPPUNIT_ASSERT(attrB2.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrB.get(i), attrB2.get(i));
            }

            filein2.close();
        }

        // Clean up temp files.
        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        AttributeArrayI attrUniform(count);

        // write out uniform attribute array to a temp file
        {
            filename = tempDir + "/openvdb_delayed2";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrUniform.writeMetadata(fileout, false, /*paged=*/true);

            compression::PagedOutputStream outputStreamSize(fileout);
            outputStreamSize.setSizeOnly(true);
            attrUniform.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout);
            outputStream.setSizeOnly(false);
            attrUniform.writePagedBuffers(outputStream, false);
            outputStream.flush();

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        proxy = new ProxyMappedFile(filename);
        mappedFile.reset(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isUniform());

            attrB.fill(5);

            CPPUNIT_ASSERT(attrB.isUniform());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(5, attrB.get(i));
            }
        }

        AttributeArrayI attrStrided(count, /*stride=*/3);

        CPPUNIT_ASSERT_EQUAL(Index(3), attrStrided.stride());

        // Clean up temp files.
        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        // write out strided attribute array to a temp file
        {
            filename = tempDir + "/openvdb_delayed3";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrStrided.writeMetadata(fileout, false, /*paged=*/true);

            compression::PagedOutputStream outputStreamSize(fileout);
            outputStreamSize.setSizeOnly(true);
            attrStrided.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout);
            outputStream.setSizeOnly(false);
            attrStrided.writePagedBuffers(outputStream, false);
            outputStream.flush();

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        proxy = new ProxyMappedFile(filename);
        mappedFile.reset(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT_EQUAL(Index(3), attrB.stride());
        }

        // Clean up temp files.
        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        // write out compressed attribute array to a temp file
        {
            filename = tempDir + "/openvdb_delayed4";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            attrA.compress();
#pragma GCC diagnostic pop
            attrA.writeMetadata(fileout, false, /*paged=*/true);

            compression::PagedOutputStream outputStreamSize(fileout);
            outputStreamSize.setSizeOnly(true);
            attrA.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout);
            outputStream.setSizeOnly(false);
            attrA.writePagedBuffers(outputStream, false);
            outputStream.flush();

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        proxy = new ProxyMappedFile(filename);
        mappedFile.reset(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check manual loading of data
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            CPPUNIT_ASSERT(!attrB.isCompressed());

            CPPUNIT_ASSERT(attrB.isOutOfCore());
            attrB.loadData();
            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            CPPUNIT_ASSERT(!attrB.isCompressed());
#pragma GCC diagnostic pop

            CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
            }
        }

        // read in using delayed load and check partial read state
        {
            std::unique_ptr<AttributeArrayI> attrB(new AttributeArrayI);

            CPPUNIT_ASSERT(!(attrB->flags() & AttributeArray::PARTIALREAD));

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB->readMetadata(filein);

            // PARTIALREAD flag should now be set
            CPPUNIT_ASSERT(attrB->flags() & AttributeArray::PARTIALREAD);

            // copy-construct and assign AttributeArray
            AttributeArrayI attrC(*attrB);
            CPPUNIT_ASSERT(attrC.flags() & AttributeArray::PARTIALREAD);
            AttributeArrayI attrD = *attrB;
            CPPUNIT_ASSERT(attrD.flags() & AttributeArray::PARTIALREAD);

            // verify deleting attrB is safe
            attrB.reset();

            // verify data is not valid
            CPPUNIT_ASSERT(!attrC.validData());

            { // attempting to write a partially-read AttributeArray throws
                std::string filename = tempDir + "/openvdb_partial1";
                std::ofstream fileout(filename.c_str(), std::ios_base::binary);
                io::setStreamMetadataPtr(fileout, streamMetadata);
                io::setDataCompression(fileout, io::COMPRESS_BLOSC);

                CPPUNIT_ASSERT_THROW(attrC.writeMetadata(fileout, false, /*paged=*/true), IoError);
            }

            // continue loading with copy-constructed AttributeArray

            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrC.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrC.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrC.isOutOfCore());
            attrC.loadData();
            CPPUNIT_ASSERT(!attrC.isOutOfCore());

            // verify data is now valid
            CPPUNIT_ASSERT(attrC.validData());

            CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrC.memUsage());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrC.get(i));
            }
        }

        // read in using delayed load and check implicit load through get()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            attrB.get(0);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
            }
        }

#ifdef OPENVDB_USE_BLOSC
        // read in using delayed load and check no implicit load through compress()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

            CPPUNIT_ASSERT(attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrB.isCompressed());

            attrB.compress();

            CPPUNIT_ASSERT(attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrB.isCompressed());

#pragma GCC diagnostic pop
        }

        // read in using delayed load and check copy and assignment constructors
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

            AttributeArrayI attrC(attrB);
            AttributeArrayI attrD = attrB;

            CPPUNIT_ASSERT(attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrC.isOutOfCore());
            CPPUNIT_ASSERT(attrD.isOutOfCore());

            attrB.loadData();
            attrC.loadData();
            attrD.loadData();

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrC.isOutOfCore());
            CPPUNIT_ASSERT(!attrD.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrC.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrD.get(i));
            }
        }

        // read in using delayed load and check implicit load through AttributeHandle
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            CPPUNIT_ASSERT(attrB.isOutOfCore());

// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            CPPUNIT_ASSERT(!attrB.isCompressed());

            AttributeHandle<int> handle(attrB);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrB.isCompressed());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), handle.get(i));
            }

            AttributeHandle<int> handle2(attrB, /*preserveCompression=*/false);
            CPPUNIT_ASSERT(!attrB.isCompressed());
#pragma GCC diagnostic pop
        }
#endif

        // Clean up temp files.
        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        // write out invalid serialization flags as metadata to a temp file
        {
            filename = tempDir + "/openvdb_delayed5";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            // write out unknown serialization flags to check forwards-compatibility

            Index64 bytes(0);
            uint8_t flags(0);
            uint8_t serializationFlags(Int16(0x10));
            Index size(0);

            fileout.write(reinterpret_cast<const char*>(&bytes), sizeof(Index64));
            fileout.write(reinterpret_cast<const char*>(&flags), sizeof(uint8_t));
            fileout.write(reinterpret_cast<const char*>(&serializationFlags), sizeof(uint8_t));
            fileout.write(reinterpret_cast<const char*>(&size), sizeof(Index));

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        proxy = new ProxyMappedFile(filename);
        mappedFile.reset(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check metadata fail due to serialization flags
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            CPPUNIT_ASSERT_THROW(attrB.readMetadata(filein), openvdb::IoError);
        }

        // cleanup temp files

        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());
    }
}


void
TestAttributeArray::testQuaternions()
{
    using AttributeQF = TypedAttributeArray<math::Quat<float>>;
    using AttributeQD = TypedAttributeArray<QuatR>;

    AttributeQF::registerType();
    AttributeQD::registerType();

    CPPUNIT_ASSERT(AttributeQF::attributeType().first == "quats");
    CPPUNIT_ASSERT(AttributeQD::attributeType().first == "quatd");

    AttributeQF test(/*size=*/5);

    AttributeQD orient(/*size=*/10);

    { // set some quaternion values
        AttributeWriteHandle<QuatR> orientHandle(orient);

        orientHandle.set(4, QuatR(1, 2, 3, 4));
        orientHandle.set(7, QuatR::identity());
    }

    { // get some quaternion values
        AttributeHandle<QuatR> orientHandle(orient);

        CPPUNIT_ASSERT_EQUAL(QuatR::zero(), orientHandle.get(3));
        CPPUNIT_ASSERT_EQUAL(QuatR(1, 2, 3, 4), orientHandle.get(4));
        CPPUNIT_ASSERT_EQUAL(QuatR::identity(), orientHandle.get(7));
    }

    { // create a quaternion array with a zero uniform value
        AttributeQD zero(/*size=*/10, /*stride=*/1, /*constantStride=*/true, QuatR::zero());

        CPPUNIT_ASSERT_EQUAL(QuatR::zero(), zero.get(5));
    }
}


void
TestAttributeArray::testMatrices()
{
    typedef TypedAttributeArray<Mat4d>      AttributeM;

    AttributeM::registerType();

    CPPUNIT_ASSERT(AttributeM::attributeType().first == "mat4d");

    AttributeM matrix(/*size=*/10);

    Mat4d testMatrix(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    { // set some matrix values
        AttributeWriteHandle<Mat4d> matrixHandle(matrix);

        matrixHandle.set(4, testMatrix);
        matrixHandle.set(7, Mat4d::zero());
    }

    { // get some matrix values
        AttributeHandle<Mat4d> matrixHandle(matrix);

        CPPUNIT_ASSERT_EQUAL(Mat4d::zero(), matrixHandle.get(3));
        CPPUNIT_ASSERT_EQUAL(testMatrix, matrixHandle.get(4));
        CPPUNIT_ASSERT_EQUAL(Mat4d::zero(), matrixHandle.get(7));
    }

    { // create a matrix array with a zero uniform value
        AttributeM zero(/*size=*/10, /*stride=*/1, /*constantStride=*/true, Mat4d::zero());

        CPPUNIT_ASSERT_EQUAL(Mat4d::zero(), zero.get(5));
    }
}


namespace profile {

template <typename AttrT>
void expand(const Name& prefix, AttrT& attr)
{
    ProfileTimer timer(prefix + ": expand");
    attr.expand();
}

template <typename AttrT>
void set(const Name& prefix, AttrT& attr)
{
    ProfileTimer timer(prefix + ": set");
    const Index size = attr.size();
    for (Index i = 0; i < size; i++) {
        attr.setUnsafe(i, typename AttrT::ValueType(i));
    }
}

template <typename CodecT, typename AttrT>
void setH(const Name& prefix, AttrT& attr)
{
    using ValueType = typename AttrT::ValueType;
    ProfileTimer timer(prefix + ": setHandle");
    AttributeWriteHandle<ValueType, CodecT> handle(attr);
    const Index size = attr.size();
    for (Index i = 0; i < size; i++) {
        handle.set(i, ValueType(i));
    }
}

template <typename AttrT>
void sum(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sum");
    typename AttrT::ValueType sum = 0;
    const Index size = attr.size();
    for (Index i = 0; i < size; i++) {
        sum += attr.getUnsafe(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

template <typename CodecT, typename AttrT>
void sumH(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sumHandle");
    using ValueType = typename AttrT::ValueType;
    ValueType sum = 0;
    AttributeHandle<ValueType, CodecT> handle(attr);
    for (Index i = 0; i < attr.size(); i++) {
        sum += handle.get(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

} // namespace profile

void
TestAttributeArray::testProfile()
{
    using namespace openvdb::util;
    using namespace openvdb::math;

    using AttributeArrayF   = TypedAttributeArray<float>;
    using AttributeArrayF16 = TypedAttributeArray<float, FixedPointCodec<false>>;
    using AttributeArrayF8  = TypedAttributeArray<float, FixedPointCodec<true>>;

    ///////////////////////////////////////////////////

#ifdef PROFILE
    const size_t elements(1000 * 1000 * 1000);

    std::cerr << std::endl;
#else
    const size_t elements(10 * 1000 * 1000);
#endif

    // std::vector

    {
        std::vector<float> values;
        {
            ProfileTimer timer("Vector<float>: resize");
            values.resize(elements);
        }
        {
            ProfileTimer timer("Vector<float>: set");
            for (size_t i = 0; i < elements; i++) {
                values[i] = float(i);
            }
        }
        {
            ProfileTimer timer("Vector<float>: sum");
            float sum = 0;
            for (size_t i = 0; i < elements; i++) {
                sum += float(values[i]);
            }
            // to prevent optimisation clean up
            CPPUNIT_ASSERT(sum);
        }
    }

    // AttributeArray

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeArray<float>", attr);
        profile::set("AttributeArray<float>", attr);
        profile::sum("AttributeArray<float>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeArray<float, fp16>", attr);
        profile::set("AttributeArray<float, fp16>", attr);
        profile::sum("AttributeArray<float, fp16>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeArray<float, fp8>", attr);
        profile::set("AttributeArray<float, fp8>", attr);
        profile::sum("AttributeArray<float, fp8>", attr);
    }

    // AttributeHandle (UnknownCodec)

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeHandle<float>", attr);
        profile::setH<UnknownCodec>("AttributeHandle<float>", attr);
        profile::sumH<UnknownCodec>("AttributeHandle<float>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeHandle<float, fp16>", attr);
        profile::setH<UnknownCodec>("AttributeHandle<float, fp16>", attr);
        profile::sumH<UnknownCodec>("AttributeHandle<float, fp16>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeHandle<float, fp8>", attr);
        profile::setH<UnknownCodec>("AttributeHandle<float, fp8>", attr);
        profile::sumH<UnknownCodec>("AttributeHandle<float, fp8>", attr);
    }

    // AttributeHandle (explicit codec)

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeHandle<float>", attr);
        profile::setH<NullCodec>("AttributeHandle<float, Codec>", attr);
        profile::sumH<NullCodec>("AttributeHandle<float, Codec>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeHandle<float, fp16>", attr);
        profile::setH<FixedPointCodec<false>>("AttributeHandle<float, fp16, Codec>", attr);
        profile::sumH<FixedPointCodec<false>>("AttributeHandle<float, fp16, Codec>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeHandle<float, fp8>", attr);
        profile::setH<FixedPointCodec<true>>("AttributeHandle<float, fp8, Codec>", attr);
        profile::sumH<FixedPointCodec<true>>("AttributeHandle<float, fp8, Codec>", attr);
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
