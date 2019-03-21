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
///
/// @file   PagedArray.h
///
/// @author Ken Museth
///
/// @brief  Concurrent, page-based, dynamically-sized linear data
///         structure with O(1) random access and STL-compliant
///         iterators. It is primarily intended for applications
///         that involve multi-threading push_back of (a possibly
///         unkown number of) elements into a dynamically growing
///         linear array, and fast random access to said elements.

#ifndef OPENVDB_UTIL_PAGED_ARRAY_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_PAGED_ARRAY_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>// std::swap
#include <tbb/atomic.h>
#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

////////////////////////////////////////


/// @brief   Concurrent, page-based, dynamically-sized linear data structure
///          with O(1) random access and STL-compliant iterators. It is
///          primarily intended for applications that concurrently insert
///          (a possibly unkown number of) elements into a dynamically
///          growing linear array, and fast random access to said elements.
///
/// @note    Multiple threads can grow the page-table and push_back
///          new elements concurrently. A ValueBuffer provides accelerated
///          and threadsafe push_back at the cost of potentially re-ordering
///          elements (when multiple instances are used).
///
/// @details This data structure employes contiguous pages of elements
///          (like a std::deque) which avoids moving data when the
///          capacity is out-grown and new pages are allocated. The
///          size of the pages can be controlled with the Log2PageSize
///          template parameter (defaults to 1024 elements of type ValueT).
///          The TableT template parameter is used to define the data
///          structure for the page table. The default, std::vector,
///          offers fast random access in exchange for slower
///          push_back, whereas std:deque offers faster push_back but
///          slower random access.
///
/// There are three fundamentally different ways to insert elements to
/// this container - each with different advanteges and disadvanteges.
///
/// The simplest way to insert elements is to use PagedArray::push_back e.g.
/// @code
///   PagedArray<size_t> array;
///   for (size_t i=0; i<100000; ++i) array.push_back(i);
/// @endcode
/// or with TBB task-based multi-threading
/// @code
///   PagedArray<size_t> array;
///   tbb::parallel_for(
///       tbb::blocked_range<size_t>(0, 10000, array.pageSize()),
///       [&array](const tbb::blocked_range<size_t>& range) {
///           for (size_t i=range.begin(); i!=range.end(); ++i) array.push_back(i);
///       }
///   );
/// @endcode
/// PagedArray::push_back has the advantage that it's thread-safe and
/// preserves the ordering of the inserted elements. In fact it returns
/// the linear offset to the added element which can then be used for
/// fast O(1) random access. The disadvantage is it's the slowest of
/// the three different ways of inserting elements.
///
/// The fastest way (by far) to insert elements by means of a PagedArray::ValueBuffer, e.g.
/// @code
///   PagedArray<size_t> array;
///   PagedArray<size_t>::ValueBuffer buffer(array);
///   for (size_t i=0; i<100000; ++i) buffer.push_back(i);
///   buffer.flush();
/// @endcode
/// or
/// @code
///   PagedArray<size_t> array;
///   {
///       //local scope of a single thread
///       PagedArray<size_t>::ValueBuffer buffer(array);
///       for (size_t i=0; i<100000; ++i) buffer.push_back(i);
///   }
/// @endcode
/// or with TBB task-based multi-threading
/// @code
///   PagedArray<size_t> array;
///   tbb::parallel_for(
///       tbb::blocked_range<size_t>(0, 10000, array.pageSize()),
///       [&array](const tbb::blocked_range<size_t>& range) {
///           PagedArray<size_t>::ValueBuffer buffer(array);
///           for (size_t i=range.begin(); i!=range.end(); ++i) buffer.push_back(i);
///       }
///   );
/// @endcode
/// or with TBB thread-local storage for even better performance (due
/// to fewer concurrent instantiations of partially full ValueBuffers)
/// @code
///   PagedArray<size_t> array;
///   PagedArray<size_t>::ValueBuffer exemplar(array);//dummy used for initialization
///   tbb::enumerable_thread_specific<PagedArray<size_t>::ValueBuffer>
///       pool(exemplar);//thread local storage pool of ValueBuffers
///   tbb::parallel_for(
///       tbb::blocked_range<size_t>(0, 10000, array.pageSize()),
///       [&pool](const tbb::blocked_range<size_t>& range) {
///           PagedArray<size_t>::ValueBuffer &buffer = pool.local();
///           for (size_t i=range.begin(); i!=range.end(); ++i) buffer.push_back(i);
///       }
///   );
///   for (auto i=pool.begin(); i!=pool.end(); ++i) i->flush();
/// @endcode
/// This technique generally outperforms PagedArray::push_back,
/// std::vector::push_back, std::deque::push_back and even
/// tbb::concurrent_vector::push_back. Additionally it
/// is thread-safe as long as each thread has it's own instance of a
/// PagedArray::ValueBuffer. The only disadvantage is the ordering of
/// the elements is undefined if multiple instance of a
/// PagedArray::ValueBuffer are employed. This is typically the case
/// in the context of multi-threading, where the
/// ordering of inserts are undefined anyway. Note that a local scope
/// can be used to guarentee that the ValueBuffer has inserted all its
/// elements by the time the scope ends. Alternatively the ValueBuffer
/// can be explicitly flushed by calling ValueBuffer::flush.
///
/// The third way to insert elements is to resize the container and use
/// random access, e.g.
/// @code
///   PagedArray<int> array;
///   array.resize(100000);
///   for (int i=0; i<100000; ++i) array[i] = i;
/// @endcode
/// or in terms of the random access iterator
/// @code
///   PagedArray<int> array;
///   array.resize(100000);
///   for (auto i=array.begin(); i!=array.end(); ++i) *i = i.pos();
/// @endcode
/// While this approach is both fast and thread-safe it suffers from the
/// major disadvantage that the problem size, i.e. number of elements, needs to
/// be known in advance. If that's the case you might as well consider
/// using std::vector or a raw c-style array! In other words the
/// PagedArray is most useful in the context of applications that
/// involve multi-threading of dynamically growing linear arrays that
/// require fast random access.

template<typename ValueT,
         size_t Log2PageSize = 10UL,
         template<typename ...> class TableT = std::vector>
class PagedArray
{
private:
    class Page;

#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1700
    // Workaround for ICC 15/16 "too few arguments to template" bug (fixed in ICC 17)
    using PageTableT = TableT<Page*, std::allocator<Page*>>;
#else
    using PageTableT = TableT<Page*>;
#endif

public:
    using ValueType = ValueT;

    /// @brief Default constructor
    PagedArray() = default;

    /// @brief Destructor removed all allocated pages
    ~PagedArray() { this->clear(); }

    // Disallow copy construction and assignment
    PagedArray(const PagedArray&) = delete;
    PagedArray& operator=(const PagedArray&) = delete;

    /// @brief Caches values into a local memory Page to improve
    ///        performance of push_back into a PagedArray.
    ///
    /// @note The ordering of inserted elements is undefined when
    ///       multiple ValueBuffers are used!
    ///
    /// @warning By design this ValueBuffer is not threadsafe so
    ///          make sure to create an instance per thread!
    class ValueBuffer;

    /// Const std-compliant iterator
    class ConstIterator;

     /// Non-const std-compliant iterator
    class Iterator;

    /// @brief   Thread safe insertion, adds a new element at
    ///          the end and increases the container size by one and
    ///          returns the linear offset for the inserted element.
    ///
    /// @param value value to be added to this PagedArray
    ///
    /// @details Constant time complexity. May allocate a new page.
    size_t push_back(const ValueType& value)
    {
        const size_t index = mSize.fetch_and_increment();
        if (index >= mCapacity) this->grow(index);
        (*mPageTable[index >> Log2PageSize])[index] = value;
        return index;
    }

    /// @brief Slightly faster than the thread-safe push_back above.
    ///
    /// @param value value to be added to this PagedArray
    ///
    /// @note For best performance consider using the ValueBuffer!
    ///
    /// @warning Not thread-safe!
    size_t push_back_unsafe(const ValueType& value)
    {
        const size_t index = mSize.fetch_and_increment();
        if (index >= mCapacity) {
            mPageTable.push_back( new Page() );
            mCapacity += Page::Size;
        }
        (*mPageTable[index >> Log2PageSize])[index] = value;
        return index;
    }

    /// @brief Reduce the page table to fix the current size.
    ///
    /// @warning Not thread-safe!
    void shrink_to_fit();

    /// @brief Return a reference to the value at the specified offset
    ///
    /// @param i linear offset of the value to be accessed.
    ///
    /// @note This random access has constant time complexity.
    ///
    /// @warning It is assumed that the i'th element is already allocated!
    ValueType& operator[](size_t i)
    {
        assert(i<mCapacity);
        return (*mPageTable[i>>Log2PageSize])[i];
    }

    /// @brief Return a const-reference to the value at the specified offset
    ///
    /// @param i linear offset of the value to be accessed.
    ///
    /// @note This random access has constant time complexity.
    ///
    /// @warning It is assumed that the i'th element is already allocated!
    const ValueType& operator[](size_t i) const
    {
        assert(i<mCapacity);
        return (*mPageTable[i>>Log2PageSize])[i];
    }

    /// @brief Set all elements in the page table to the specified value
    ///
    /// @param v value to be filled in all the existing pages of this PagedArray.
    ///
    /// @note Multi-threaded
    void fill(const ValueType& v)
    {
        auto op = [&](const tbb::blocked_range<size_t>& r){
            for(size_t i=r.begin(); i!=r.end(); ++i) mPageTable[i]->fill(v);
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->pageCount()), op);
    }

    /// @brief Copy the first @a count values in this PageArray into
    /// a raw c-style array, assuming it to be at least @a count
    /// elements long.
    ///
    /// @param p pointer to an array that will used as the destination of the copy.
    /// @param count number of elements to be copied.
    ///
    bool copy(ValueType *p, size_t count) const
    {
        size_t last_page = count >> Log2PageSize;
        if (last_page >= this->pageCount()) return false;
        auto op = [&](const tbb::blocked_range<size_t>& r){
            for (size_t i=r.begin(); i!=r.end(); ++i) {
                mPageTable[i]->copy(p+i*Page::Size, Page::Size);
            }
        };
        if (size_t m = count & Page::Mask) {//count is not divisible by page size
            tbb::parallel_for(tbb::blocked_range<size_t>(0, last_page, 32), op);
            mPageTable[last_page]->copy(p+last_page*Page::Size, m);
        } else {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, last_page+1, 32), op);
        }
        return true;
    }
    void copy(ValueType *p) const { this->copy(p, mSize); }

    /// @brief Resize this array to the specified size.
    ///
    /// @param size number of elements that this PageArray will contain.
    ///
    /// @details Will grow or shrink the page table to contain
    /// the specified number of elements. It will affect the size(),
    /// iteration will go over all those elements, push_back will
    /// insert after them and operator[] can be used directly access
    /// them.
    ///
    /// @note No reserve method is implemented due to efficiency concerns
    /// (especially for the ValueBuffer) from having to deal with empty pages.
    ///
    /// @warning Not thread-safe!
    void resize(size_t size)
    {
        mSize = size;
        if (size > mCapacity) {
            this->grow(size-1);
        } else {
            this->shrink_to_fit();
        }
    }

    /// @brief Resize this array to the specified size and initialize
    /// all values to @a v.
    ///
    /// @param size number of elements that this PageArray will contain.
    /// @param v value of all the @a size values.
    ///
    /// @details Will grow or shrink the page table to contain
    /// the specified number of elements. It will affect the size(),
    /// iteration will go over all those elements, push_back will
    /// insert after them and operator[] can be used directly access them.
    ///
    /// @note No reserve method is implemented due to efficiency concerns
    /// (especially for the ValueBuffer) from having to deal with empty pages.
    ///
    /// @warning Not thread-safe!
    void resize(size_t size, const ValueType& v)
    {
       this->resize(size);
       this->fill(v);
    }

    /// @brief Return the number of elements in this array.
    size_t size() const { return mSize; }

    /// @brief Return the maximum number of elements that this array
    /// can contain without allocating more memory pages.
    size_t capacity() const { return mCapacity; }

    /// @brief Return the number of additional elements that can be
    /// added to this array without allocating more memory pages.
    size_t freeCount() const { return mCapacity - mSize; }

    /// @brief Return the number of allocated memory pages.
    size_t pageCount() const { return mPageTable.size(); }

    /// @brief Return the number of elements per memory page.
    static size_t pageSize() { return Page::Size; }

    /// @brief Return log2 of the number of elements per memory page.
    static size_t log2PageSize() { return Log2PageSize; }

    /// @brief Return the memory footprint of this array in bytes.
    size_t memUsage() const
    {
        return sizeof(*this) + mPageTable.size() * Page::memUsage();
    }

    /// @brief Return true if the container contains no elements.
    bool isEmpty() const { return mSize == 0; }

    /// @brief Return true if the page table is partially full, i.e. the
    ///        last non-empty page contains less than pageSize() elements.
    ///
    /// @details When the page table is partially full calling merge()
    ///          or using a ValueBuffer will rearrange the ordering of
    ///          existing elements.
    bool isPartiallyFull() const { return (mSize & Page::Mask) > 0; }

    /// @brief  Removes all elements from the array and delete all pages.
    ///
    /// @warning Not thread-safe!
    void clear()
    {
        for (size_t i=0, n=mPageTable.size(); i<n; ++i) delete mPageTable[i];
        PageTableT().swap(mPageTable);
        mSize     = 0;
        mCapacity = 0;
    }

    /// @brief Return a non-const iterator pointing to the first element
    Iterator begin() { return Iterator(*this, 0); }

    /// @brief Return a non-const iterator pointing to the
    /// past-the-last element.
    ///
    /// @warning Iterator does not point to a valid element and should not
    /// be dereferenced!
    Iterator end() { return Iterator(*this, mSize); }

    //@{
    /// @brief Return a const iterator pointing to the first element
    ConstIterator cbegin() const { return ConstIterator(*this, 0); }
    ConstIterator begin() const { return ConstIterator(*this, 0); }
    //@}

    //@{
    /// @brief Return a const iterator pointing to the
    /// past-the-last element.
    ///
    /// @warning Iterator does not point to a valid element and should not
    /// be dereferenced!
    ConstIterator cend() const { return ConstIterator(*this, mSize); }
    ConstIterator end() const { return ConstIterator(*this, mSize); }
    //@}

    /// @brief Parallel sort of all the elements in ascending order.
    void sort() { tbb::parallel_sort(this->begin(), this->end(), std::less<ValueT>() ); }

    /// @brief Parallel sort of all the elements in descending order.
    void invSort() { tbb::parallel_sort(this->begin(), this->end(), std::greater<ValueT>()); }

    //@{
    /// @brief Parallel sort of all the elements based on a custom
    /// functor with the api:
    /// @code bool operator()(const ValueT& a, const ValueT& b) @endcode
    /// which returns true if a comes before b.
    template <typename Functor>
    void sort(Functor func) { tbb::parallel_sort(this->begin(), this->end(), func ); }
    //@}

    /// @brief Transfer all the elements (and pages) from the other array to this array.
    ///
    /// @param other non-const reference to the PagedArray that will be merged into this PagedArray.
    ///
    /// @note The other PagedArray is empty on return.
    ///
    /// @warning The ordering of elements is undefined if this page table is partially full!
    void merge(PagedArray& other);

    /// @brief Print information for debugging
    void print(std::ostream& os = std::cout) const
      {
          os << "PagedArray:\n"
             << "\tSize:       " << this->size() << " elements\n"
             << "\tPage table: " << this->pageCount() << " pages\n"
             << "\tPage size:  " << this->pageSize() << " elements\n"
             << "\tCapacity:   " << this->capacity() << " elements\n"
             << "\tFootprint:  " << this->memUsage() << " bytes\n";
      }

private:

    friend class ValueBuffer;

    void grow(size_t index)
    {
        tbb::spin_mutex::scoped_lock lock(mGrowthMutex);
        while(index >= mCapacity) {
            mPageTable.push_back( new Page() );
            mCapacity += Page::Size;
        }
    }

    void add_full(Page*& page, size_t size);

    void add_partially_full(Page*& page, size_t size);

    void add(Page*& page, size_t size) {
        tbb::spin_mutex::scoped_lock lock(mGrowthMutex);
        if (size == Page::Size) {//page is full
            this->add_full(page, size);
        } else if (size>0) {//page is only partially full
            this->add_partially_full(page, size);
        }
    }
    PageTableT mPageTable;//holds points to allocated pages
    tbb::atomic<size_t> mSize{0};// current number of elements in array
    size_t mCapacity = 0;//capacity of array given the current page count
    tbb::spin_mutex mGrowthMutex;//Mutex-lock required to grow pages
}; // Public class PagedArray

////////////////////////////////////////////////////////////////////////////////

template <typename ValueT, size_t Log2PageSize, template<typename ...> class TableT>
void PagedArray<ValueT, Log2PageSize, TableT>::shrink_to_fit()
{
    if (mPageTable.size() > (mSize >> Log2PageSize) + 1) {
        tbb::spin_mutex::scoped_lock lock(mGrowthMutex);
        const size_t pageCount = (mSize >> Log2PageSize) + 1;
        if (mPageTable.size() > pageCount) {
            delete mPageTable.back();
            mPageTable.pop_back();
            mCapacity -= Page::Size;
        }
    }
}

template <typename ValueT, size_t Log2PageSize, template<typename ...> class TableT>
void PagedArray<ValueT, Log2PageSize,TableT >::merge(PagedArray& other)
{
    if (&other != this && !other.isEmpty()) {
        tbb::spin_mutex::scoped_lock lock(mGrowthMutex);
        // extract last partially full page if it exists
        Page* page = nullptr;
        const size_t size = mSize & Page::Mask; //number of elements in the last page
        if ( size > 0 ) {
            page = mPageTable.back();
            mPageTable.pop_back();
            mSize -= size;
        }
        // transfer all pages from the other page table
        mPageTable.insert(mPageTable.end(), other.mPageTable.begin(), other.mPageTable.end());
        mSize          += other.mSize;
        mCapacity       = Page::Size*mPageTable.size();
        other.mSize     = 0;
        other.mCapacity = 0;
        PageTableT().swap(other.mPageTable);
        // add back last partially full page
        if (page) this->add_partially_full(page, size);
    }
}

template <typename ValueT, size_t Log2PageSize, template<typename ...> class TableT>
void PagedArray<ValueT, Log2PageSize, TableT>::add_full(Page*& page, size_t size)
{
    assert(size == Page::Size);//page must be full
    if (mSize & Page::Mask) {//page-table is partially full
        Page*& tmp = mPageTable.back();
        std::swap(tmp, page);//swap last table entry with page
    }
    mPageTable.push_back(page);
    mCapacity += Page::Size;
    mSize     += size;
    page       = nullptr;
}

template <typename ValueT, size_t Log2PageSize, template<typename ...> class TableT>
void PagedArray<ValueT, Log2PageSize, TableT>::add_partially_full(Page*& page, size_t size)
{
    assert(size > 0 && size < Page::Size);//page must be partially full
    if (size_t m = mSize & Page::Mask) {//page table is also partially full
        ValueT *s = page->data(), *t = mPageTable.back()->data() + m;
        for (size_t i=std::min(mSize+size, mCapacity)-mSize; i; --i) *t++ = *s++;
        if (mSize+size > mCapacity) {//grow page table
            mPageTable.push_back( new Page() );
            t = mPageTable.back()->data();
            for (size_t i=mSize+size-mCapacity; i; --i) *t++ = *s++;
            mCapacity += Page::Size;
        }
    } else {//page table is full so simply append page
        mPageTable.push_back( page );
        mCapacity += Page::Size;
        page       = nullptr;
    }
    mSize += size;
}

////////////////////////////////////////////////////////////////////////////////

// Public member-class of PagedArray
template <typename ValueT, size_t Log2PageSize, template<typename ...> class TableT>
class PagedArray<ValueT, Log2PageSize, TableT>::
ValueBuffer
{
public:
    using PagedArrayType = PagedArray<ValueT, Log2PageSize, TableT>;
    /// @brief Constructor from a PageArray
    ValueBuffer(PagedArray& parent) : mParent(&parent), mPage(new Page()), mSize(0) {}
    /// @warning This copy-constructor is shallow in the sense that no
    ///          elements are copied, i.e. size = 0.
    ValueBuffer(const ValueBuffer& other) : mParent(other.mParent), mPage(new Page()), mSize(0) {}
    /// @brief Destructor that transfers an buffered values to the parent PagedArray.
    ~ValueBuffer() { mParent->add(mPage, mSize); delete mPage; }

    ValueBuffer& operator=(const ValueBuffer&) = delete;// disallow copy assignment

    /// @brief Add a value to the buffer and increment the size.
    ///
    /// @details If the internal memory page is full it will
    /// automaically flush the page to the parent PagedArray.
    void push_back(const ValueT& v) {
        (*mPage)[mSize++] = v;
        if (mSize == Page::Size) this->flush();
    }
    /// @brief Manually transfers the values in this buffer to the parent PagedArray.
    ///
    /// @note This method is also called by the destructor and
    /// push_back so it should only be called if one manually wants to
    /// sync up the buffer with the array, e.g. during debugging.
    void flush() {
        mParent->add(mPage, mSize);
        if (mPage == nullptr) mPage = new Page();
        mSize = 0;
    }
    /// @brief Return a reference to the parent PagedArray
    PagedArrayType& parent() const { return *mParent; }
    /// @brief Return the current number of elements cached in this buffer.
    size_t size() const { return mSize; }
private:
    PagedArray* mParent;
    Page*       mPage;
    size_t      mSize;
};// Public class PagedArray::ValueBuffer

////////////////////////////////////////////////////////////////////////////////

// Const std-compliant iterator
// Public member-class of PagedArray
template <typename ValueT, size_t Log2PageSize, template<typename ...> class TableT>
class PagedArray<ValueT, Log2PageSize, TableT>::
ConstIterator : public std::iterator<std::random_access_iterator_tag, ValueT>
{
public:
    using BaseT = std::iterator<std::random_access_iterator_tag, ValueT>;
    using difference_type = typename BaseT::difference_type;
    // constructors and assignment
    ConstIterator() : mPos(0), mParent(nullptr) {}
    ConstIterator(const PagedArray& parent, size_t pos=0) : mPos(pos), mParent(&parent) {}
    ConstIterator(const ConstIterator& other) : mPos(other.mPos), mParent(other.mParent) {}
    ConstIterator& operator=(const ConstIterator& other) {
        mPos=other.mPos;
        mParent=other.mParent;
        return *this;
    }
    // prefix
    ConstIterator& operator++() { ++mPos; return *this; }
    ConstIterator& operator--() { --mPos; return *this; }
    // postfix
    ConstIterator  operator++(int) { ConstIterator tmp(*this); ++mPos; return tmp; }
    ConstIterator  operator--(int) { ConstIterator tmp(*this); --mPos; return tmp; }
    // value access
    const ValueT& operator*()  const { return (*mParent)[mPos]; }
    const ValueT* operator->() const { return &(this->operator*()); }
    const ValueT& operator[](const difference_type& pos) const { return (*mParent)[mPos+pos]; }
    // offset
    ConstIterator& operator+=(const difference_type& pos) { mPos += pos; return *this; }
    ConstIterator& operator-=(const difference_type& pos) { mPos -= pos; return *this; }
    ConstIterator operator+(const difference_type &pos) const { return Iterator(*mParent,mPos+pos); }
    ConstIterator operator-(const difference_type &pos) const { return Iterator(*mParent,mPos-pos); }
    difference_type operator-(const ConstIterator& other) const { return mPos - other.pos(); }
    // comparisons
    bool operator==(const ConstIterator& other) const { return mPos == other.mPos; }
    bool operator!=(const ConstIterator& other) const { return mPos != other.mPos; }
    bool operator>=(const ConstIterator& other) const { return mPos >= other.mPos; }
    bool operator<=(const ConstIterator& other) const { return mPos <= other.mPos; }
    bool operator< (const ConstIterator& other) const { return mPos <  other.mPos; }
    bool operator> (const ConstIterator& other) const { return mPos >  other.mPos; }
    // non-std methods
    bool isValid() const { return mParent != nullptr && mPos < mParent->size(); }
    size_t pos()   const { return mPos; }
private:
    size_t            mPos;
    const PagedArray* mParent;
};// Public class PagedArray::ConstIterator

////////////////////////////////////////////////////////////////////////////////

// Non-const std-compliant iterator
// Public member-class of PagedArray
template <typename ValueT, size_t Log2PageSize, template<typename ...> class TableT>
class PagedArray<ValueT, Log2PageSize, TableT>::
Iterator : public std::iterator<std::random_access_iterator_tag, ValueT>
{
public:
    using BaseT = std::iterator<std::random_access_iterator_tag, ValueT>;
    using difference_type = typename BaseT::difference_type;
    // constructors and assignment
    Iterator() : mPos(0), mParent(nullptr) {}
    Iterator(PagedArray& parent, size_t pos=0) : mPos(pos), mParent(&parent) {}
    Iterator(const Iterator& other) : mPos(other.mPos), mParent(other.mParent) {}
    Iterator& operator=(const Iterator& other) {
        mPos=other.mPos;
        mParent=other.mParent;
        return *this;
    }
    // prefix
    Iterator& operator++() { ++mPos; return *this; }
    Iterator& operator--() { --mPos; return *this; }
    // postfix
    Iterator  operator++(int) { Iterator tmp(*this); ++mPos; return tmp; }
    Iterator  operator--(int) { Iterator tmp(*this); --mPos; return tmp; }
    // value access
    ValueT& operator*()  const { return (*mParent)[mPos]; }
    ValueT* operator->() const { return &(this->operator*()); }
    ValueT& operator[](const difference_type& pos) const { return (*mParent)[mPos+pos]; }
    // offset
    Iterator& operator+=(const difference_type& pos) { mPos += pos; return *this; }
    Iterator& operator-=(const difference_type& pos) { mPos -= pos; return *this; }
    Iterator operator+(const difference_type &pos) const { return Iterator(*mParent, mPos+pos); }
    Iterator operator-(const difference_type &pos) const { return Iterator(*mParent, mPos-pos); }
    difference_type operator-(const Iterator& other) const { return mPos - other.pos(); }
    // comparisons
    bool operator==(const Iterator& other) const { return mPos == other.mPos; }
    bool operator!=(const Iterator& other) const { return mPos != other.mPos; }
    bool operator>=(const Iterator& other) const { return mPos >= other.mPos; }
    bool operator<=(const Iterator& other) const { return mPos <= other.mPos; }
    bool operator< (const Iterator& other) const { return mPos <  other.mPos; }
    bool operator> (const Iterator& other) const { return mPos >  other.mPos; }
    // non-std methods
    bool isValid() const { return mParent != nullptr && mPos < mParent->size(); }
    size_t pos()   const { return mPos; }
  private:
    size_t      mPos;
    PagedArray* mParent;
};// Public class PagedArray::Iterator

////////////////////////////////////////////////////////////////////////////////

// Private member-class of PagedArray implementing a memory page
template <typename ValueT, size_t Log2PageSize, template<typename ...> class TableT>
class PagedArray<ValueT, Log2PageSize, TableT>::
Page
{
public:
    static const size_t Size = 1UL << Log2PageSize;
    static const size_t Mask = Size - 1UL;
    static size_t memUsage() { return sizeof(ValueT)*Size; }
    // Raw memory allocation without any initialization
    Page() : mData(reinterpret_cast<ValueT*>(new char[sizeof(ValueT)*Size])) {}
    ~Page() { delete [] mData; }
    Page(const Page&) = delete;//copy construction is not implemented
    Page& operator=(const Page&) = delete;//copy assignment is not implemented
    ValueT& operator[](const size_t i) { return mData[i & Mask]; }
    const ValueT& operator[](const size_t i) const { return mData[i & Mask]; }
    void fill(const ValueT& v) {
        ValueT* dst = mData;
        for (size_t i=Size; i; --i) *dst++ = v;
    }
    ValueT* data() { return mData; }
    // Copy the first n elements of this Page to dst (which is assumed to large
    // enough to hold the n elements).
    void copy(ValueType *dst, size_t n) const {
        const ValueT* src = mData;
        for (size_t i=n; i; --i) *dst++ = *src++;
    }
protected:
    ValueT* mData;
};// Private class PagedArray::Page

////////////////////////////////////////////////////////////////////////////////

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_PAGED_ARRAY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
