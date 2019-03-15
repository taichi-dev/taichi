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

#include <tbb/tbb_thread.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <openvdb/Exceptions.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/util/PagedArray.h>

#include <iostream>

//#define BENCHMARK_PAGED_ARRAY

// For benchmark comparisons
#ifdef BENCHMARK_PAGED_ARRAY
#include <deque> // for std::deque
#include <vector> // for std::vector
#include <tbb/tbb.h> // for tbb::concurrent_vector
#endif

class TestUtil: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestUtil);
    CPPUNIT_TEST(testCpuTimer);
    CPPUNIT_TEST(testPagedArray);
    CPPUNIT_TEST_SUITE_END();

    void testCpuTimer();
    void testPagedArray();

    using RangeT = tbb::blocked_range<size_t>;

    // Multi-threading ArrayT::push_back
    template<typename ArrayT>
    struct ArrayPushBack {
        ArrayPushBack(ArrayT& array) : mArray(&array) {}
        void parallel(size_t size) {tbb::parallel_for(RangeT(size_t(0), size, mArray->pageSize()), *this);}
        void serial(size_t size) { (*this)(RangeT(size_t(0), size)); }
        void unsafe(size_t size) { for (size_t i=0; i!=size; ++i) mArray->push_back_unsafe(i); }
        void operator()(const RangeT& r) const {
            for (size_t i=r.begin(), n=r.end(); i!=n; ++i) mArray->push_back(i);
        }
        ArrayT* mArray;
    };

    // Multi-threading ArrayT::ValueBuffer::push_back
    template<typename ArrayT>
    struct BufferPushBack {
        BufferPushBack(ArrayT& array) : mBuffer(array) {}
        void parallel(size_t size) {
            tbb::parallel_for(RangeT(size_t(0), size, mBuffer.parent().pageSize()), *this);
        }
        void serial(size_t size) { (*this)(RangeT(size_t(0), size)); }
        void operator()(const RangeT& r) const {
            for (size_t i=r.begin(), n=r.end(); i!=n; ++i) mBuffer.push_back(i);
        }
        mutable typename ArrayT::ValueBuffer mBuffer;//local instance
    };

    // Thread Local Storage version of BufferPushBack
    template<typename ArrayT>
    struct TLS_BufferPushBack {
        using PoolT = tbb::enumerable_thread_specific<typename ArrayT::ValueBuffer>;
        TLS_BufferPushBack(ArrayT &array) : mArray(&array), mPool(nullptr) {}
        void parallel(size_t size) {
            typename ArrayT::ValueBuffer exemplar(*mArray);//dummy used for initialization
            mPool = new PoolT(exemplar);//thread local storage pool of ValueBuffers
            tbb::parallel_for(RangeT(size_t(0), size, mArray->pageSize()), *this);
            for (auto i=mPool->begin(); i!=mPool->end(); ++i) i->flush();
            delete mPool;
        }
        void operator()(const RangeT& r) const {
            typename PoolT::reference buffer = mPool->local();
            for (size_t i=r.begin(), n=r.end(); i!=n; ++i) buffer.push_back(i);
        }
        ArrayT *mArray;
        PoolT  *mPool;
    };
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestUtil);


void
TestUtil::testCpuTimer()
{
    const int expected = 259, tolerance = 20;//milliseconds
    const tbb::tick_count::interval_t sec(expected/1000.0);

    openvdb::util::CpuTimer timer;
    tbb::this_tbb_thread::sleep(sec);
    const int actual1 = static_cast<int>(timer.delta());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual1, tolerance);
    tbb::this_tbb_thread::sleep(sec);
    const int actual2 = static_cast<int>(timer.delta());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2*expected, actual2, tolerance);
}


void
TestUtil::testPagedArray()
{
#ifdef BENCHMARK_PAGED_ARRAY
    const size_t problemSize = 2560000;
    openvdb::util::CpuTimer timer;
    std::cerr << "\nProblem size for benchmark: " << problemSize << std::endl;
#else
    const size_t problemSize = 256000;
#endif

    {//serial PagedArray::push_back (check return value)
        openvdb::util::PagedArray<int, size_t(8)> d;
        CPPUNIT_ASSERT(d.isEmpty());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(8), d.log2PageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.capacity());

        CPPUNIT_ASSERT_EQUAL(size_t(0), d.push_back(10));
        CPPUNIT_ASSERT_EQUAL(10, d[0]);
        CPPUNIT_ASSERT(!d.isEmpty());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.capacity());

        CPPUNIT_ASSERT_EQUAL(size_t(1), d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(size_t(2), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.capacity());

        for (size_t i=2; i<d.pageSize(); ++i) CPPUNIT_ASSERT_EQUAL(i, d.push_back(int(i)));
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.capacity());

        for (int i=2, n=int(d.size()); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

        CPPUNIT_ASSERT_EQUAL(d.pageSize(), d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(d.pageSize()+1, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(2), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(2*d.pageSize(), d.capacity());
    }
    {//serial PagedArray::push_back
        using ArrayT = openvdb::util::PagedArray<size_t>;
        ArrayT d;
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("1: Serial PagedArray::push_back with default page size");
#endif
        {// for some reason this:
            ArrayPushBack<ArrayT> tmp(d);
            tmp.serial(problemSize);
        }// is faster than:
        //for (size_t i=0; i<problemSize; ++i) d.push_back(i);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(size_t(10), d.log2PageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//serial PagedArray::push_back_unsafe
        using ArrayT = openvdb::util::PagedArray<size_t>;
        ArrayT d;
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("2: Serial PagedArray::push_back_unsafe with default page size");
#endif
        {// for some reason this:
            ArrayPushBack<ArrayT> tmp(d);
            tmp.unsafe(problemSize);
        }// is faster than:
        //openvdb::util::PagedArray<size_t> d;
        //for (size_t i=0; i<problemSize; ++i) d.push_back_unsafe(i);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//parallel PagedArray::push_back
        using ArrayT = openvdb::util::PagedArray<size_t>;
        ArrayT d;
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("3: Parallel PagedArray::push_back with default page size");
#endif
        //{// for some reason this:
        //    ArrayPushBack<ArrayT> tmp(d);
        //    tmp.parallel(problemSize);
        //}// is faster than:
        tbb::parallel_for(tbb::blocked_range<size_t>(0, problemSize, d.pageSize()),
                          [&d](const tbb::blocked_range<size_t> &range){
                          for (size_t i=range.begin(); i!=range.end(); ++i) d.push_back(i);});
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(size_t(10), d.log2PageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort with a default page size");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

        CPPUNIT_ASSERT_EQUAL(problemSize, d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(problemSize+1, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());


    }
    {//parallel PagedArray::push_back with page size of only 8
        using ArrayT = openvdb::util::PagedArray<size_t, size_t(3)>;
        ArrayT d;
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("4: Parallel PagedArray::push_back with page size of only 8");
#endif
        {// for some reason this:
            ArrayPushBack<ArrayT> tmp(d);
            tmp.parallel(problemSize);
        }// is faster than:
        //tbb::parallel_for(tbb::blocked_range<size_t>(0, problemSize, d.pageSize()),
        //                  [&d](const tbb::blocked_range<size_t> &range){
        //                  for (size_t i=range.begin(); i!=range.end(); ++i) d.push_back(i);});
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(size_t(3), d.log2PageSize());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort with a page size of only 8");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

        CPPUNIT_ASSERT_EQUAL(problemSize, d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(problemSize+1, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
    }
#ifdef BENCHMARK_PAGED_ARRAY
    {//benchmark against a std::vector
        timer.start("5: Serial std::vector::push_back");
        std::vector<size_t> v;
        for (size_t i=0; i<problemSize; ++i) v.push_back(i);
        timer.stop();
        CPPUNIT_ASSERT_EQUAL(problemSize, v.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, v[i]);
    }
    {//benchmark against a std::deque
        timer.start("6: Serial std::deque::push_back");
        std::deque<size_t> d;
        for (size_t i=0; i<problemSize; ++i) d.push_back(i);
        timer.stop();
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());

        std::deque<int> d2;
        CPPUNIT_ASSERT_EQUAL(size_t(0), d2.size());
        d2.resize(1234);
        CPPUNIT_ASSERT_EQUAL(size_t(1234), d2.size());
    }
    {//benchmark against a tbb::concurrent_vector::push_back
        timer.start("7: Serial tbb::concurrent_vector::push_back");
        tbb::concurrent_vector<size_t> v;
        for (size_t i=0; i<problemSize; ++i) v.push_back(i);
        timer.stop();
        CPPUNIT_ASSERT_EQUAL(problemSize, v.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, v[i]);

        v.clear();
        timer.start("8: Parallel tbb::concurrent_vector::push_back");
        using ArrayT = openvdb::util::PagedArray<size_t>;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, problemSize, ArrayT::pageSize()),
                          [&v](const tbb::blocked_range<size_t> &range){
                          for (size_t i=range.begin(); i!=range.end(); ++i) v.push_back(i);});
        timer.stop();
        tbb::parallel_sort(v.begin(), v.end());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, v[i]);
    }
#endif

    {//serial PagedArray::ValueBuffer::push_back
        using ArrayT = openvdb::util::PagedArray<size_t>;
        ArrayT d;

        CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
        d.resize(problemSize);
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL((problemSize-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        d.clear();
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("9: Serial PagedArray::ValueBuffer::push_back");
#endif
        {// for some reason this:
            BufferPushBack<ArrayT> tmp(d);
            tmp.serial(problemSize);
            // is faster than:
            //ArrayT::ValueBuffer buffer(d);
            //for (size_t i=0, n=problemSize; i<n; ++i) buffer.push_back(i);
        }

#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        for (size_t i=0; i<problemSize; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

        size_t unsorted = 0;
        for (size_t i=0, n=d.size(); i<n; ++i) unsorted += i != d[i];
        CPPUNIT_ASSERT_EQUAL(size_t(0), unsorted);

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);


        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());


    }
    {//parallel PagedArray::ValueBuffer::push_back
        using ArrayT = openvdb::util::PagedArray<size_t>;
        ArrayT d;
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("10: Parallel PagedArray::ValueBuffer::push_back");
#endif
        {// for some reason this:
            BufferPushBack<ArrayT> tmp(d);
            tmp.parallel(problemSize);
        }// is faster than:
        //tbb::parallel_for(tbb::blocked_range<size_t>(0, problemSize, d.pageSize()),
        //                  [&d](const tbb::blocked_range<size_t> &r){
        //                  typename ArrayT::ValueBuffer buffer(d);
        //                  for (size_t i=r.begin(), n=r.end(); i!=n; ++i) buffer.push_back(i);});
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif

        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        // Test sorting
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0; i<d.size(); ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel inverse sort");
#endif
        d.invSort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size()-1; i<=n; ++i) CPPUNIT_ASSERT_EQUAL(n-i, d[i]);

        CPPUNIT_ASSERT_EQUAL(problemSize, d.push_back(1));
        CPPUNIT_ASSERT_EQUAL(problemSize+1, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        // pageCount - 1 = max index >> log2PageSize
        CPPUNIT_ASSERT_EQUAL(size_t(1)+(problemSize>>d.log2PageSize()), d.pageCount());
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        // test PagedArray::fill
        const size_t v = 13;
        d.fill(v);
        for (size_t i=0, n=d.capacity(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(v, d[i]);
    }
    {//test PagedArray::ValueBuffer::flush
        using ArrayT = openvdb::util::PagedArray<size_t>;
        ArrayT d;
        CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
        {
            ArrayT::ValueBuffer vc(d);
            vc.push_back(1);
            vc.push_back(2);
            CPPUNIT_ASSERT_EQUAL(size_t(0), d.size());
            vc.flush();
            CPPUNIT_ASSERT_EQUAL(size_t(2), d.size());
            CPPUNIT_ASSERT_EQUAL(size_t(1), d[0]);
            CPPUNIT_ASSERT_EQUAL(size_t(2), d[1]);
        }
        CPPUNIT_ASSERT_EQUAL(size_t(2), d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1), d[0]);
        CPPUNIT_ASSERT_EQUAL(size_t(2), d[1]);
    }
    {//thread-local-storage PagedArray::ValueBuffer::push_back followed by parallel sort
        using ArrayT = openvdb::util::PagedArray<size_t>;
        ArrayT d;

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("11: Parallel TLS PagedArray::ValueBuffer::push_back");
#endif
        {// for some reason this:
            TLS_BufferPushBack<ArrayT> tmp(d);
            tmp.parallel(problemSize);
        }// is faster than:
        //ArrayT::ValueBuffer exemplar(d);//dummy used for initialization
        ///tbb::enumerable_thread_specific<ArrayT::ValueBuffer> pool(exemplar);//thread local storage pool of ValueBuffers
        //tbb::parallel_for(tbb::blocked_range<size_t>(0, problemSize, d.pageSize()),
        //                  [&pool](const tbb::blocked_range<size_t> &range){
        //                  ArrayT::ValueBuffer &buffer = pool.local();
        //                  for (size_t i=range.begin(); i!=range.end(); ++i) buffer.push_back(i);});
        //for (auto i=pool.begin(); i!=pool.end(); ++i) i->flush();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        //std::cerr << "Number of threads for TLS = " << (buffer.end()-buffer.begin()) << std::endl;
        //d.print();
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());

        // Not guaranteed to pass
        //size_t unsorted = 0;
        //for (size_t i=0, n=d.size(); i<n; ++i) unsorted += i != d[i];
        //CPPUNIT_ASSERT( unsorted > 0 );

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//parallel PagedArray::merge followed by parallel sort
        using ArrayT = openvdb::util::PagedArray<size_t>;
        ArrayT d, d2;

        tbb::parallel_for(tbb::blocked_range<size_t>(0, problemSize, d.pageSize()),
                          [&d](const tbb::blocked_range<size_t> &range){
                          ArrayT::ValueBuffer buffer(d);
                          for (size_t i=range.begin(); i!=range.end(); ++i) buffer.push_back(i);});
        CPPUNIT_ASSERT_EQUAL(problemSize, d.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d.log2PageSize(), d.pageSize());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d.pageCount()*d.pageSize(), d.capacity());
        CPPUNIT_ASSERT(!d.isPartiallyFull());
        d.push_back(problemSize);
        CPPUNIT_ASSERT(d.isPartiallyFull());

        tbb::parallel_for(tbb::blocked_range<size_t>(problemSize+1, 2*problemSize+1, d2.pageSize()),
                          [&d2](const tbb::blocked_range<size_t> &range){
                          ArrayT::ValueBuffer buffer(d2);
                          for (size_t i=range.begin(); i!=range.end(); ++i) buffer.push_back(i);});
        //for (size_t i=d.size(), n=i+problemSize; i<n; ++i) d2.push_back(i);
        CPPUNIT_ASSERT(!d2.isPartiallyFull());
        CPPUNIT_ASSERT_EQUAL(problemSize, d2.size());
        CPPUNIT_ASSERT_EQUAL(size_t(1)<<d2.log2PageSize(), d2.pageSize());
        CPPUNIT_ASSERT_EQUAL((d2.size()-1)>>d2.log2PageSize(), d2.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(d2.pageCount()*d2.pageSize(), d2.capacity());

        //d.print();
        //d2.print();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel PagedArray::merge");
#endif
        d.merge(d2);
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        CPPUNIT_ASSERT(d.isPartiallyFull());

        //d.print();
        //d2.print();
        CPPUNIT_ASSERT_EQUAL(2*problemSize+1, d.size());
        CPPUNIT_ASSERT_EQUAL((d.size()-1)>>d.log2PageSize(), d.pageCount()-1);
        CPPUNIT_ASSERT_EQUAL(size_t(0), d2.size());
        CPPUNIT_ASSERT_EQUAL(size_t(0), d2.pageCount());

#ifdef BENCHMARK_PAGED_ARRAY
        timer.start("parallel sort of merged array");
#endif
        d.sort();
#ifdef BENCHMARK_PAGED_ARRAY
        timer.stop();
#endif
        for (size_t i=0, n=d.size(); i<n; ++i) CPPUNIT_ASSERT_EQUAL(i, d[i]);
    }
    {//examples in doxygen
        {//1
            openvdb::util::PagedArray<int> array;
            for (int i=0; i<100000; ++i) array.push_back(i);
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
        {//2A
            openvdb::util::PagedArray<int> array;
            openvdb::util::PagedArray<int>::ValueBuffer buffer(array);
            for (int i=0; i<100000; ++i) buffer.push_back(i);
            buffer.flush();
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
        {//2B
            openvdb::util::PagedArray<int> array;
            {//local scope of a single thread
                openvdb::util::PagedArray<int>::ValueBuffer buffer(array);
                for (int i=0; i<100000; ++i) buffer.push_back(i);
            }
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
        {//3A
            openvdb::util::PagedArray<int> array;
            array.resize(100000);
            for (int i=0; i<100000; ++i) array[i] = i;
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
        {//3B
            using ArrayT = openvdb::util::PagedArray<int>;
            ArrayT array;
            array.resize(100000);
            for (ArrayT::Iterator i=array.begin(); i!=array.end(); ++i) *i = int(i.pos());
            for (int i=0; i<100000; ++i) CPPUNIT_ASSERT_EQUAL(i, array[i]);
        }
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
