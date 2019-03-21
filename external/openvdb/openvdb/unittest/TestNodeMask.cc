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
#include <openvdb/Exceptions.h>
#include <openvdb/util/NodeMasks.h>

using openvdb::Index;

template<typename MaskType> void TestAll();

class TestNodeMask: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestNodeMask);
    CPPUNIT_TEST(testAll4);
    CPPUNIT_TEST(testAll3);
    CPPUNIT_TEST(testAll2);
    CPPUNIT_TEST(testAll1);
    CPPUNIT_TEST_SUITE_END();

    void testAll4() { TestAll<openvdb::util::NodeMask<4> >(); }
    void testAll3() { TestAll<openvdb::util::NodeMask<3> >(); }
    void testAll2() { TestAll<openvdb::util::NodeMask<2> >(); }
    void testAll1() { TestAll<openvdb::util::NodeMask<1> >(); }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestNodeMask);

template<typename MaskType>
void TestAll()
{
    CPPUNIT_ASSERT(MaskType::memUsage() == MaskType::SIZE/8);
    const Index SIZE = MaskType::SIZE > 512 ? 512 : MaskType::SIZE;

    {// default constructor
        MaskType m;//all bits are off
        for (Index i=0; i<SIZE; ++i) CPPUNIT_ASSERT(m.isOff(i));
        for (Index i=0; i<SIZE; ++i) CPPUNIT_ASSERT(!m.isOn(i));
        CPPUNIT_ASSERT(m.isOff());
        CPPUNIT_ASSERT(!m.isOn());
        CPPUNIT_ASSERT(m.countOn() == 0);
        CPPUNIT_ASSERT(m.countOff()== MaskType::SIZE);
        m.toggle();//all bits are on
        CPPUNIT_ASSERT(m.isOn());
        CPPUNIT_ASSERT(!m.isOff());
        CPPUNIT_ASSERT(m.countOn() == MaskType::SIZE);
        CPPUNIT_ASSERT(m.countOff()== 0);
        for (Index i=0; i<SIZE; ++i) CPPUNIT_ASSERT(!m.isOff(i));
        for (Index i=0; i<SIZE; ++i) CPPUNIT_ASSERT(m.isOn(i));
    }
    {// On constructor
        MaskType m(true);//all bits are on
        CPPUNIT_ASSERT(m.isOn());
        CPPUNIT_ASSERT(!m.isOff());
        CPPUNIT_ASSERT(m.countOn() == MaskType::SIZE);
        CPPUNIT_ASSERT(m.countOff()== 0);
        for (Index i=0; i<SIZE; ++i) CPPUNIT_ASSERT(!m.isOff(i));
        for (Index i=0; i<SIZE; ++i) CPPUNIT_ASSERT(m.isOn(i));
        m.toggle();
        for (Index i=0; i<SIZE; ++i) CPPUNIT_ASSERT(m.isOff(i));
        for (Index i=0; i<SIZE; ++i) CPPUNIT_ASSERT(!m.isOn(i));
        CPPUNIT_ASSERT(m.isOff());
        CPPUNIT_ASSERT(!m.isOn());
        CPPUNIT_ASSERT(m.countOn() == 0);
        CPPUNIT_ASSERT(m.countOff()== MaskType::SIZE);
    }
    {// Off constructor
        MaskType m(false);
        CPPUNIT_ASSERT(m.isOff());
        CPPUNIT_ASSERT(!m.isOn());
        CPPUNIT_ASSERT(m.countOn() == 0);
        CPPUNIT_ASSERT(m.countOff()== MaskType::SIZE);
        m.setOn();
        CPPUNIT_ASSERT(m.isOn());
        CPPUNIT_ASSERT(!m.isOff());
        CPPUNIT_ASSERT(m.countOn() == MaskType::SIZE);
        CPPUNIT_ASSERT(m.countOff()== 0);
        m = MaskType();//copy asignment
        CPPUNIT_ASSERT(m.isOff());
        CPPUNIT_ASSERT(!m.isOn());
        CPPUNIT_ASSERT(m.countOn() == 0);
        CPPUNIT_ASSERT(m.countOff()== MaskType::SIZE);
    }
    {// test setOn, setOff, findFirstOn and findFiratOff
        MaskType m;
        for (Index i=0; i<SIZE; ++i) {
            m.setOn(i);
            CPPUNIT_ASSERT(m.countOn() == 1);
            CPPUNIT_ASSERT(m.findFirstOn() == i);
            CPPUNIT_ASSERT(m.findFirstOff() == (i==0 ? 1 : 0));
            for (Index j=0; j<SIZE; ++j) {
                CPPUNIT_ASSERT( i==j ? m.isOn(j) : m.isOff(j) );
            }
            m.setOff(i);
            CPPUNIT_ASSERT(m.countOn() == 0);
            CPPUNIT_ASSERT(m.findFirstOn() == MaskType::SIZE);
        }
    }
    {// OnIterator
        MaskType m;
        for (Index i=0; i<SIZE; ++i) {
            m.setOn(i);
            for (typename MaskType::OnIterator iter=m.beginOn(); iter; ++iter) {
                CPPUNIT_ASSERT( iter.pos() == i );
            }
            CPPUNIT_ASSERT(m.countOn() == 1);
            m.setOff(i);
            CPPUNIT_ASSERT(m.countOn() == 0);
        }
    }
    {// OffIterator
        MaskType m(true);
        for (Index i=0; i<SIZE; ++i) {
            m.setOff(i);
            CPPUNIT_ASSERT(m.countOff() == 1);
            for (typename MaskType::OffIterator iter=m.beginOff(); iter; ++iter) {
                CPPUNIT_ASSERT( iter.pos() == i );
            }
            CPPUNIT_ASSERT(m.countOn() == MaskType::SIZE-1);
            m.setOn(i);
            CPPUNIT_ASSERT(m.countOff() == 0);
            CPPUNIT_ASSERT(m.countOn() == MaskType::SIZE);
        }
    }
    {// isConstant
        MaskType m(true);//all bits are on
        bool isOn = false;
        CPPUNIT_ASSERT(!m.isOff());
        CPPUNIT_ASSERT(m.isOn());
        CPPUNIT_ASSERT(m.isConstant(isOn));
        CPPUNIT_ASSERT(isOn);
        m.setOff(MaskType::SIZE-1);//sets last bit off
        CPPUNIT_ASSERT(!m.isOff());
        CPPUNIT_ASSERT(!m.isOn());
        CPPUNIT_ASSERT(!m.isConstant(isOn));
        m.setOff();//sets all bits off
        CPPUNIT_ASSERT(m.isOff());
        CPPUNIT_ASSERT(!m.isOn());
        CPPUNIT_ASSERT(m.isConstant(isOn));
        CPPUNIT_ASSERT(!isOn);
    }
    {// DenseIterator
        MaskType m(false);
        for (Index i=0; i<SIZE; ++i) {
            m.setOn(i);
            CPPUNIT_ASSERT(m.countOn() == 1);
            for (typename MaskType::DenseIterator iter=m.beginDense(); iter; ++iter) {
                CPPUNIT_ASSERT( iter.pos()==i ? *iter : !*iter );
            }
            m.setOff(i);
            CPPUNIT_ASSERT(m.countOn() == 0);
        }
    }
}




// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
