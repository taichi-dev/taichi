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
//
/// @author Ken Museth
///
/// @file NodeMasks.h

#ifndef OPENVDB_UTIL_NODEMASKS_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_NODEMASKS_HAS_BEEN_INCLUDED

#include <algorithm> // for std::min()
#include <cassert>
#include <cstring>
#include <iostream>// for cout
#include <openvdb/Platform.h>
#include <openvdb/Types.h>
//#include <boost/mpl/if.hpp>
//#include <strings.h> // for ffs


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

/// Return the number of on bits in the given 8-bit value.
inline Index32
CountOn(Byte v)
{
    // Simple LUT:
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
    static
#endif
    /// @todo Move this table and others into, say, Util.cc
    const Byte numBits[256] = {
#define COUNTONB2(n)  n,            n+1,            n+1,            n+2
#define COUNTONB4(n)  COUNTONB2(n), COUNTONB2(n+1), COUNTONB2(n+1), COUNTONB2(n+2)
#define COUNTONB6(n)  COUNTONB4(n), COUNTONB4(n+1), COUNTONB4(n+1), COUNTONB4(n+2)
        COUNTONB6(0), COUNTONB6(1), COUNTONB6(1),   COUNTONB6(2)
    };
    return numBits[v];
#undef COUNTONB6
#undef COUNTONB4
#undef COUNTONB2

    // Sequentially clear least significant bits
    //Index32 c;
    //for (c = 0; v; c++)  v &= v - 0x01U;
    //return c;

    // This version is only fast on CPUs with fast "%" and "*" operations
    //return (v * UINT64_C(0x200040008001) & UINT64_C(0x111111111111111)) % 0xF;
}

/// Return the number of off bits in the given 8-bit value.
inline Index32 CountOff(Byte v) { return CountOn(static_cast<Byte>(~v)); }

/// Return the number of on bits in the given 32-bit value.
inline Index32
CountOn(Index32 v)
{
    v = v - ((v >> 1) & 0x55555555U);
    v = (v & 0x33333333U) + ((v >> 2) & 0x33333333U);
    return (((v + (v >> 4)) & 0xF0F0F0FU) * 0x1010101U) >> 24;
}

/// Return the number of off bits in the given 32-bit value.
inline Index32 CountOff(Index32 v) { return CountOn(~v); }

/// Return the number of on bits in the given 64-bit value.
inline Index32
CountOn(Index64 v)
{
    v = v - ((v >> 1) & UINT64_C(0x5555555555555555));
    v = (v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333));
    return static_cast<Index32>(
        (((v + (v >> 4)) & UINT64_C(0xF0F0F0F0F0F0F0F)) * UINT64_C(0x101010101010101)) >> 56);
}

/// Return the number of off bits in the given 64-bit value.
inline Index32 CountOff(Index64 v) { return CountOn(~v); }

/// Return the least significant on bit of the given 8-bit value.
inline Index32
FindLowestOn(Byte v)
{
    assert(v);
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
    static
#endif
    const Byte DeBruijn[8] = {0, 1, 6, 2, 7, 5, 4, 3};
    return DeBruijn[Byte((v & -v) * 0x1DU) >> 5];
}

/// Return the least significant on bit of the given 32-bit value.
inline Index32
FindLowestOn(Index32 v)
{
    assert(v);
    //return ffs(v);
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
    static
#endif
    const Byte DeBruijn[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };
    return DeBruijn[Index32((v & -v) * 0x077CB531U) >> 27];
}

/// Return the least significant on bit of the given 64-bit value.
inline Index32
FindLowestOn(Index64 v)
{
    assert(v);
    //return ffsll(v);
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
    static
#endif
    const Byte DeBruijn[64] = {
        0,   1,  2, 53,  3,  7, 54, 27, 4,  38, 41,  8, 34, 55, 48, 28,
        62,  5, 39, 46, 44, 42, 22,  9, 24, 35, 59, 56, 49, 18, 29, 11,
        63, 52,  6, 26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
        51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
    };
    return DeBruijn[Index64((v & -v) * UINT64_C(0x022FDD63CC95386D)) >> 58];
}

/// Return the most significant on bit of the given 32-bit value.
inline Index32
FindHighestOn(Index32 v)
{
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
    static
#endif
    const Byte DeBruijn[32] = {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
        8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
    };
    v |= v >> 1; // first round down to one less than a power of 2
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return DeBruijn[Index32(v * 0x07C4ACDDU) >> 27];
}


////////////////////////////////////////


/// Base class for the bit mask iterators
template<typename NodeMask>
class BaseMaskIterator
{
protected:
    Index32 mPos; // bit position
    const NodeMask* mParent; // this iterator can't change the parent_mask!

public:
    BaseMaskIterator(): mPos(NodeMask::SIZE), mParent(nullptr) {}
    BaseMaskIterator(const BaseMaskIterator&) = default;
    BaseMaskIterator(Index32 pos, const NodeMask* parent): mPos(pos), mParent(parent)
    {
        assert((parent == nullptr && pos == 0) || (parent != nullptr && pos <= NodeMask::SIZE));
    }
    bool operator==(const BaseMaskIterator &iter) const {return mPos == iter.mPos;}
    bool operator!=(const BaseMaskIterator &iter) const {return mPos != iter.mPos;}
    bool operator< (const BaseMaskIterator &iter) const {return mPos <  iter.mPos;}
    BaseMaskIterator& operator=(const BaseMaskIterator& iter)
    {
        mPos = iter.mPos; mParent = iter.mParent; return *this;
    }
    Index32 offset() const { return mPos; }
    Index32 pos() const { return mPos; }
    bool test() const { assert(mPos <= NodeMask::SIZE); return (mPos != NodeMask::SIZE); }
    operator bool() const { return this->test(); }
}; // class BaseMaskIterator


/// @note This happens to be a const-iterator!
template <typename NodeMask>
class OnMaskIterator: public BaseMaskIterator<NodeMask>
{
private:
    using BaseType = BaseMaskIterator<NodeMask>;
    using BaseType::mPos;//bit position;
    using BaseType::mParent;//this iterator can't change the parent_mask!
public:
    OnMaskIterator() : BaseType() {}
    OnMaskIterator(Index32 pos,const NodeMask *parent) : BaseType(pos,parent) {}
    void increment()
    {
        assert(mParent != nullptr);
        mPos = mParent->findNextOn(mPos+1);
        assert(mPos <= NodeMask::SIZE);
    }
    void increment(Index n) { while(n-- && this->next()) ; }
    bool next()
    {
        this->increment();
        return this->test();
    }
    bool operator*() const {return true;}
    OnMaskIterator& operator++()
    {
        this->increment();
        return *this;
    }
}; // class OnMaskIterator


template <typename NodeMask>
class OffMaskIterator: public BaseMaskIterator<NodeMask>
{
private:
    using BaseType = BaseMaskIterator<NodeMask>;
    using BaseType::mPos;//bit position;
    using BaseType::mParent;//this iterator can't change the parent_mask!
public:
    OffMaskIterator() : BaseType()  {}
    OffMaskIterator(Index32 pos,const NodeMask *parent) : BaseType(pos,parent) {}
    void increment()
    {
        assert(mParent != nullptr);
        mPos=mParent->findNextOff(mPos+1);
        assert(mPos <= NodeMask::SIZE);
    }
    void increment(Index n) { while(n-- && this->next()) ; }
    bool next()
    {
        this->increment();
        return this->test();
    }
    bool operator*() const {return false;}
    OffMaskIterator& operator++()
    {
        this->increment();
        return *this;
    }
}; // class OffMaskIterator


template <typename NodeMask>
class DenseMaskIterator: public BaseMaskIterator<NodeMask>
{
private:
    using BaseType = BaseMaskIterator<NodeMask>;
    using BaseType::mPos;//bit position;
    using BaseType::mParent;//this iterator can't change the parent_mask!

public:
    DenseMaskIterator() : BaseType() {}
    DenseMaskIterator(Index32 pos,const NodeMask *parent) : BaseType(pos,parent) {}
    void increment()
    {
        assert(mParent != nullptr);
        mPos += 1;//careful - the increment might go beyond the end
        assert(mPos<= NodeMask::SIZE);
    }
    void increment(Index n) { while(n-- && this->next()) ; }
    bool next()
    {
        this->increment();
        return this->test();
    }
    bool operator*() const {return mParent->isOn(mPos);}
    DenseMaskIterator& operator++()
    {
        this->increment();
        return *this;
    }
}; // class DenseMaskIterator


/// @brief Bit mask for the internal and leaf nodes of VDB. This
/// is a 64-bit implementation.
///
/// @note A template specialization for Log2Dim=1 and Log2Dim=2 are
/// given below.
template<Index Log2Dim>
class NodeMask
{
public:
    static_assert(Log2Dim > 2, "expected NodeMask template specialization, got base template");

    static const Index32 LOG2DIM    = Log2Dim;
    static const Index32 DIM        = 1<<Log2Dim;
    static const Index32 SIZE       = 1<<3*Log2Dim;
    static const Index32 WORD_COUNT = SIZE >> 6;// 2^6=64
    using Word = Index64;

private:

    // The bits are represented as a linear array of Words, and the
    // size of a Word is 32 or 64 bits depending on the platform.
    // The BIT_MASK is defined as the number of bits in a Word - 1
    //static const Index32 BIT_MASK   = sizeof(void*) == 8 ? 63 : 31;
    //static const Index32 LOG2WORD   = BIT_MASK == 63 ? 6 : 5;
    //static const Index32 WORD_COUNT = SIZE >> LOG2WORD;
    //using Word = boost::mpl::if_c<BIT_MASK == 63, Index64, Index32>::type;

    Word mWords[WORD_COUNT];//only member data!

public:
    /// Default constructor sets all bits off
    NodeMask() { this->setOff(); }
    /// All bits are set to the specified state
     NodeMask(bool on) { this->set(on); }
    /// Copy constructor
    NodeMask(const NodeMask &other) { *this = other; }
    /// Destructor
    ~NodeMask() {}
    /// Assignment operator
    NodeMask& operator=(const NodeMask& other)
    {
        Index32 n = WORD_COUNT;
        const Word* w2 = other.mWords;
        for (Word* w1 = mWords; n--; ++w1, ++w2) *w1 = *w2;
        return *this;
    }

    using OnIterator = OnMaskIterator<NodeMask>;
    using OffIterator = OffMaskIterator<NodeMask>;
    using DenseIterator = DenseMaskIterator<NodeMask>;

    OnIterator beginOn() const       { return OnIterator(this->findFirstOn(),this); }
    OnIterator endOn() const         { return OnIterator(SIZE,this); }
    OffIterator beginOff() const     { return OffIterator(this->findFirstOff(),this); }
    OffIterator endOff() const       { return OffIterator(SIZE,this); }
    DenseIterator beginDense() const { return DenseIterator(0,this); }
    DenseIterator endDense() const   { return DenseIterator(SIZE,this); }

    bool operator == (const NodeMask &other) const
    {
        int n = WORD_COUNT;
        for (const Word *w1=mWords, *w2=other.mWords; n-- && *w1++ == *w2++;) ;
        return n == -1;
    }

    bool operator != (const NodeMask &other) const { return !(*this == other); }

    //
    // Bitwise logical operations
    //

    /// @brief Apply a functor to the words of the this and the other mask.
    ///
    /// @details An example that implements the "operator&=" method:
    /// @code
    /// struct Op { inline void operator()(W &w1, const W& w2) const { w1 &= w2; } };
    /// @endcode
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other, const WordOp& op)
    {
        Word *w1 = mWords;
        const Word *w2 = other.mWords;
        for (Index32 n = WORD_COUNT; n--;  ++w1, ++w2) op( *w1, *w2);
        return *this;
    }
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other1, const NodeMask& other2, const WordOp& op)
    {
        Word *w1 = mWords;
        const Word *w2 = other1.mWords, *w3 = other2.mWords;
        for (Index32 n = WORD_COUNT; n--;  ++w1, ++w2, ++w3) op( *w1, *w2, *w3);
        return *this;
    }
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other1, const NodeMask& other2, const NodeMask& other3,
                            const WordOp& op)
    {
        Word *w1 = mWords;
        const Word *w2 = other1.mWords, *w3 = other2.mWords, *w4 = other3.mWords;
        for (Index32 n = WORD_COUNT; n--;  ++w1, ++w2, ++w3, ++w4) op( *w1, *w2, *w3, *w4);
        return *this;
    }
    /// @brief Bitwise intersection
    const NodeMask& operator&=(const NodeMask& other)
    {
        Word *w1 = mWords;
        const Word *w2 = other.mWords;
        for (Index32 n = WORD_COUNT; n--;  ++w1, ++w2) *w1 &= *w2;
        return *this;
    }
    /// @brief Bitwise union
    const NodeMask& operator|=(const NodeMask& other)
    {
        Word *w1 = mWords;
        const Word *w2 = other.mWords;
        for (Index32 n = WORD_COUNT; n--;  ++w1, ++w2) *w1 |= *w2;
        return *this;
    }
    /// @brief Bitwise difference
    const NodeMask& operator-=(const NodeMask& other)
    {
        Word *w1 = mWords;
        const Word *w2 = other.mWords;
        for (Index32 n = WORD_COUNT; n--;  ++w1, ++w2) *w1 &= ~*w2;
        return *this;
    }
    /// @brief Bitwise XOR
    const NodeMask& operator^=(const NodeMask& other)
    {
        Word *w1 = mWords;
        const Word *w2 = other.mWords;
        for (Index32 n = WORD_COUNT; n--;  ++w1, ++w2) *w1 ^= *w2;
        return *this;
    }
    NodeMask operator!()                      const { NodeMask m(*this); m.toggle(); return m; }
    NodeMask operator&(const NodeMask& other) const { NodeMask m(*this); m &= other; return m; }
    NodeMask operator|(const NodeMask& other) const { NodeMask m(*this); m |= other; return m; }
    NodeMask operator^(const NodeMask& other) const { NodeMask m(*this); m ^= other; return m; }

    /// Return the byte size of this NodeMask
    static Index32 memUsage() { return static_cast<Index32>(WORD_COUNT*sizeof(Word)); }
    /// Return the total number of on bits
    Index32 countOn() const
    {
        Index32 sum = 0, n = WORD_COUNT;
        for (const Word* w = mWords; n--; ++w) sum += CountOn(*w);
        return sum;
    }
    /// Return the total number of on bits
    Index32 countOff() const { return SIZE-this->countOn(); }
    /// Set the <i>n</i>th  bit on
    void setOn(Index32 n) {
        assert( (n >> 6) < WORD_COUNT );
        mWords[n >> 6] |=  Word(1) << (n & 63);
    }
    /// Set the <i>n</i>th bit off
    void setOff(Index32 n) {
        assert( (n >> 6) < WORD_COUNT );
        mWords[n >> 6] &=  ~(Word(1) << (n & 63));
    }
    /// Set the <i>n</i>th bit to the specified state
    void set(Index32 n, bool On) { On ? this->setOn(n) : this->setOff(n); }
    /// Set all bits to the specified state
    void set(bool on)
    {
        const Word state = on ? ~Word(0) : Word(0);
        Index32 n = WORD_COUNT;
        for (Word* w = mWords; n--; ++w) *w = state;
    }
    /// Set all bits on
    void setOn()
    {
        Index32 n = WORD_COUNT;
        for (Word* w = mWords; n--; ++w) *w = ~Word(0);
    }
    /// Set all bits off
    void setOff()
    {
        Index32 n = WORD_COUNT;
        for (Word* w = mWords; n--; ++w) *w = Word(0);
    }
    /// Toggle the state of the <i>n</i>th bit
    void toggle(Index32 n) {
        assert( (n >> 6) < WORD_COUNT );
        mWords[n >> 6] ^= Word(1) << (n & 63);
    }
    /// Toggle the state of all bits in the mask
    void toggle()
    {
        Index32 n = WORD_COUNT;
        for (Word* w = mWords; n--; ++w) *w = ~*w;
    }
    /// Set the first bit on
    void setFirstOn()  { this->setOn(0); }
    /// Set the last bit on
    void setLastOn()   { this->setOn(SIZE-1); }
    /// Set the first bit off
    void setFirstOff() { this->setOff(0); }
    /// Set the last bit off
    void setLastOff()  { this->setOff(SIZE-1); }
    /// Return @c true if the <i>n</i>th bit is on
    bool isOn(Index32 n) const
    {
        assert( (n >> 6) < WORD_COUNT );
        return 0 != (mWords[n >> 6] & (Word(1) << (n & 63)));
    }
    /// Return @c true if the <i>n</i>th bit is off
    bool isOff(Index32 n) const {return !this->isOn(n); }
    /// Return @c true if all the bits are on
    bool isOn() const
    {
        int n = WORD_COUNT;
        for (const Word *w = mWords; n-- && *w++ == ~Word(0);) ;
        return n == -1;
    }
    /// Return @c true if all the bits are off
    bool isOff() const
    {
        int n = WORD_COUNT;
        for (const Word *w = mWords; n-- && *w++ == Word(0);) ;
        return n == -1;
    }
    /// Return @c true if bits are either all off OR all on.
    /// @param isOn Takes on the values of all bits if the method
    /// returns true - else it is undefined.
    bool isConstant(bool &isOn) const
    {
        isOn = (mWords[0] == ~Word(0));//first word has all bits on
        if ( !isOn && mWords[0] != Word(0)) return false;//early out
        const Word *w = mWords + 1, *n = mWords + WORD_COUNT;
        while( w<n && *w == mWords[0] ) ++w;
        return w == n;
    }
    Index32 findFirstOn() const
    {
        Index32 n = 0;
        const Word* w = mWords;
        for (; n<WORD_COUNT && !*w; ++w, ++n) ;
        return n==WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(*w);
    }
    Index32 findFirstOff() const
    {
        Index32 n = 0;
        const Word* w = mWords;
        for (; n<WORD_COUNT && !~*w; ++w, ++n) ;
        return n==WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(~*w);
    }

    //@{
    /// Return the <i>n</i>th word of the bit mask, for a word of arbitrary size.
    template<typename WordT>
    WordT getWord(Index n) const
    {
        assert(n*8*sizeof(WordT) < SIZE);
        return reinterpret_cast<const WordT*>(mWords)[n];
    }
    template<typename WordT>
    WordT& getWord(Index n)
    {
        assert(n*8*sizeof(WordT) < SIZE);
        return reinterpret_cast<WordT*>(mWords)[n];
    }
    //@}

    void save(std::ostream& os) const
    {
        os.write(reinterpret_cast<const char*>(mWords), this->memUsage());
    }
    void load(std::istream& is) { is.read(reinterpret_cast<char*>(mWords), this->memUsage()); }
    void seek(std::istream& is) const { is.seekg(this->memUsage(), std::ios_base::cur); }
    /// @brief simple print method for debugging
    void printInfo(std::ostream& os=std::cout) const
    {
        os << "NodeMask: Dim=" << DIM << " Log2Dim=" << Log2Dim
            << " Bit count=" << SIZE << " word count=" << WORD_COUNT << std::endl;
    }
    void printBits(std::ostream& os=std::cout, Index32 max_out=80u) const
    {
        const Index32 n=(SIZE>max_out ? max_out : SIZE);
        for (Index32 i=0; i < n; ++i) {
            if ( !(i & 63) )
                os << "||";
            else if ( !(i%8) )
                os << "|";
            os << this->isOn(i);
        }
        os << "|" << std::endl;
    }
    void printAll(std::ostream& os=std::cout, Index32 max_out=80u) const
    {
        this->printInfo(os);
        this->printBits(os, max_out);
    }

    Index32 findNextOn(Index32 start) const
    {
        Index32 n = start >> 6;//initiate
        if (n >= WORD_COUNT) return SIZE; // check for out of bounds
        Index32 m = start & 63;
        Word b = mWords[n];
        if (b & (Word(1) << m)) return start;//simpel case: start is on
        b &= ~Word(0) << m;// mask out lower bits
        while(!b && ++n<WORD_COUNT) b = mWords[n];// find next none-zero word
        return (!b ? SIZE : (n << 6) + FindLowestOn(b));//catch last word=0
    }

    Index32 findNextOff(Index32 start) const
    {
        Index32 n = start >> 6;//initiate
        if (n >= WORD_COUNT) return SIZE; // check for out of bounds
        Index32 m = start & 63;
        Word b = ~mWords[n];
        if (b & (Word(1) << m)) return start;//simpel case: start is on
        b &= ~Word(0) << m;// mask out lower bits
        while(!b && ++n<WORD_COUNT) b = ~mWords[n];// find next none-zero word
        return (!b ? SIZE : (n << 6) + FindLowestOn(b));//catch last word=0
    }
};// NodeMask


/// @brief Template specialization of NodeMask for Log2Dim=1, i.e. 2^3 nodes
template<>
class NodeMask<1>
{
public:

    static const Index32 LOG2DIM    = 1;
    static const Index32 DIM        = 2;
    static const Index32 SIZE       = 8;
    static const Index32 WORD_COUNT = 1;
    using Word = Byte;

private:

    Byte mByte;//only member data!

public:
    /// Default constructor sets all bits off
    NodeMask() : mByte(0x00U) {}
    /// All bits are set to the specified state
    NodeMask(bool on) : mByte(on ? 0xFFU : 0x00U) {}
    /// Copy constructor
    NodeMask(const NodeMask &other) : mByte(other.mByte) {}
    /// Destructor
    ~NodeMask() {}
    /// Assignment operator
    void operator = (const NodeMask &other) { mByte = other.mByte; }

    using OnIterator = OnMaskIterator<NodeMask>;
    using OffIterator = OffMaskIterator<NodeMask>;
    using DenseIterator = DenseMaskIterator<NodeMask>;

    OnIterator beginOn() const       { return OnIterator(this->findFirstOn(),this); }
    OnIterator endOn() const         { return OnIterator(SIZE,this); }
    OffIterator beginOff() const     { return OffIterator(this->findFirstOff(),this); }
    OffIterator endOff() const       { return OffIterator(SIZE,this); }
    DenseIterator beginDense() const { return DenseIterator(0,this); }
    DenseIterator endDense() const   { return DenseIterator(SIZE,this); }

    bool operator == (const NodeMask &other) const { return mByte == other.mByte; }

    bool operator != (const NodeMask &other) const {return mByte != other.mByte; }

    //
    // Bitwise logical operations
    //

    /// @brief Apply a functor to the words of the this and the other mask.
    ///
    /// @details An example that implements the "operator&=" method:
    /// @code
    /// struct Op { inline void operator()(Word &w1, const Word& w2) const { w1 &= w2; } };
    /// @endcode
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other, const WordOp& op)
    {
        op(mByte, other.mByte);
        return *this;
    }
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other1, const NodeMask& other2, const WordOp& op)
    {
        op(mByte, other1.mByte, other2.mByte);
        return *this;
    }
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other1, const NodeMask& other2, const NodeMask& other3,
                            const WordOp& op)
    {
        op(mByte, other1.mByte, other2.mByte, other3.mByte);
        return *this;
    }
    /// @brief Bitwise intersection
    const NodeMask& operator&=(const NodeMask& other)
    {
        mByte &= other.mByte;
        return *this;
    }
    /// @brief Bitwise union
    const NodeMask& operator|=(const NodeMask& other)
    {
        mByte |= other.mByte;
        return *this;
    }
    /// @brief Bitwise difference
    const NodeMask& operator-=(const NodeMask& other)
    {
        mByte &= static_cast<Byte>(~other.mByte);
        return *this;
    }
    /// @brief Bitwise XOR
    const NodeMask& operator^=(const NodeMask& other)
    {
        mByte ^= other.mByte;
        return *this;
    }
    NodeMask operator!()                      const { NodeMask m(*this); m.toggle(); return m; }
    NodeMask operator&(const NodeMask& other) const { NodeMask m(*this); m &= other; return m; }
    NodeMask operator|(const NodeMask& other) const { NodeMask m(*this); m |= other; return m; }
    NodeMask operator^(const NodeMask& other) const { NodeMask m(*this); m ^= other; return m; }
    /// Return the byte size of this NodeMask
    static Index32 memUsage() { return 1; }
    /// Return the total number of on bits
    Index32 countOn() const { return CountOn(mByte); }
    ///  Return the total number of on bits
    Index32 countOff() const { return CountOff(mByte); }
    /// Set the <i>n</i>th  bit on
    void setOn(Index32 n) {
        assert( n  < 8 );
        mByte = mByte | static_cast<Byte>(0x01U << (n & 7));
    }
    /// Set the <i>n</i>th bit off
    void setOff(Index32 n) {
        assert( n  < 8 );
        mByte = mByte & static_cast<Byte>(~(0x01U << (n & 7)));
    }
    /// Set the <i>n</i>th bit to the specified state
    void set(Index32 n, bool On) { On ? this->setOn(n) : this->setOff(n); }
    /// Set all bits to the specified state
    void set(bool on) { mByte = on ? 0xFFU : 0x00U; }
    /// Set all bits on
    void setOn() { mByte = 0xFFU; }
    /// Set all bits off
    void setOff() { mByte = 0x00U; }
    /// Toggle the state of the <i>n</i>th bit
    void toggle(Index32 n) {
        assert( n  < 8 );
        mByte = mByte ^ static_cast<Byte>(0x01U << (n & 7));
    }
    /// Toggle the state of all bits in the mask
    void toggle() { mByte = static_cast<Byte>(~mByte); }
    /// Set the first bit on
    void setFirstOn()  { this->setOn(0); }
    /// Set the last bit on
    void setLastOn()   { this->setOn(7); }
    /// Set the first bit off
    void setFirstOff() { this->setOff(0); }
    /// Set the last bit off
    void setLastOff()  { this->setOff(7); }
    /// Return true if the <i>n</i>th bit is on
    bool isOn(Index32 n) const
    {
        assert( n  < 8 );
        return mByte & (0x01U << (n & 7));
    }
    /// Return true if the <i>n</i>th bit is off
    bool isOff(Index32 n) const {return !this->isOn(n); }
    /// Return true if all the bits are on
    bool isOn() const { return mByte == 0xFFU; }
    /// Return true if all the bits are off
    bool isOff() const { return mByte == 0; }
    /// Return @c true if bits are either all off OR all on.
    /// @param isOn Takes on the values of all bits if the method
    /// returns true - else it is undefined.
    bool isConstant(bool &isOn) const
    {
        isOn = this->isOn();
        return isOn || this->isOff();
    }
    Index32 findFirstOn() const { return mByte ? FindLowestOn(mByte) : 8; }
    Index32 findFirstOff() const
    {
        const Byte b = static_cast<Byte>(~mByte);
        return b ? FindLowestOn(b) : 8;
    }
    /*
    //@{
    /// Return the <i>n</i>th word of the bit mask, for a word of arbitrary size.
    /// @note This version assumes WordT=Byte and n=0!
    template<typename WordT>
    WordT getWord(Index n) const
    {
        static_assert(sizeof(WordT) == sizeof(Byte), "expected word size to be one byte");
        assert(n == 0);
        return reinterpret_cast<WordT>(mByte);
    }
    template<typename WordT>
    WordT& getWord(Index n)
    {
        static_assert(sizeof(WordT) == sizeof(Byte), "expected word size to be one byte");
        assert(n == 0);
        return reinterpret_cast<WordT&>(mByte);
    }
    //@}
    */
    void save(std::ostream& os) const { os.write(reinterpret_cast<const char*>(&mByte), 1); }
    void load(std::istream& is) { is.read(reinterpret_cast<char*>(&mByte), 1); }
    void seek(std::istream& is) const { is.seekg(1, std::ios_base::cur); }
    /// @brief simple print method for debugging
    void printInfo(std::ostream& os=std::cout) const
    {
        os << "NodeMask: Dim=2, Log2Dim=1, Bit count=8, Word count=1"<<std::endl;
    }
    void printBits(std::ostream& os=std::cout) const
    {
        os << "||";
        for (Index32 i=0; i < 8; ++i) os << this->isOn(i);
        os << "||" << std::endl;
    }
    void printAll(std::ostream& os=std::cout) const
    {
        this->printInfo(os);
        this->printBits(os);
    }

    Index32 findNextOn(Index32 start) const
    {
        if (start>=8) return 8;
        const Byte b = static_cast<Byte>(mByte & (0xFFU << start));
        return  b ? FindLowestOn(b) : 8;
    }

    Index32 findNextOff(Index32 start) const
    {
        if (start>=8) return 8;
        const Byte b = static_cast<Byte>(~mByte & (0xFFU << start));
        return  b ? FindLowestOn(b) : 8;
    }

};// NodeMask<1>


/// @brief Template specialization of NodeMask for Log2Dim=2, i.e. 4^3 nodes
template<>
class NodeMask<2>
{
public:

    static const Index32 LOG2DIM    =  2;
    static const Index32 DIM        =  4;
    static const Index32 SIZE       = 64;
    static const Index32 WORD_COUNT = 1;
    using Word = Index64;

private:

    Word mWord;//only member data!

public:
    /// Default constructor sets all bits off
    NodeMask() : mWord(UINT64_C(0x00)) {}
    /// All bits are set to the specified state
    NodeMask(bool on) : mWord(on ? UINT64_C(0xFFFFFFFFFFFFFFFF) : UINT64_C(0x00)) {}
    /// Copy constructor
    NodeMask(const NodeMask &other) : mWord(other.mWord) {}
    /// Destructor
    ~NodeMask() {}
    /// Assignment operator
    void operator = (const NodeMask &other) { mWord = other.mWord; }

    using OnIterator = OnMaskIterator<NodeMask>;
    using OffIterator = OffMaskIterator<NodeMask>;
    using DenseIterator = DenseMaskIterator<NodeMask>;

    OnIterator beginOn() const       { return OnIterator(this->findFirstOn(),this); }
    OnIterator endOn() const         { return OnIterator(SIZE,this); }
    OffIterator beginOff() const     { return OffIterator(this->findFirstOff(),this); }
    OffIterator endOff() const       { return OffIterator(SIZE,this); }
    DenseIterator beginDense() const { return DenseIterator(0,this); }
    DenseIterator endDense() const   { return DenseIterator(SIZE,this); }

    bool operator == (const NodeMask &other) const { return mWord == other.mWord; }

    bool operator != (const NodeMask &other) const {return mWord != other.mWord; }

    //
    // Bitwise logical operations
    //

    /// @brief Apply a functor to the words of the this and the other mask.
    ///
    /// @details An example that implements the "operator&=" method:
    /// @code
    /// struct Op { inline void operator()(Word &w1, const Word& w2) const { w1 &= w2; } };
    /// @endcode
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other, const WordOp& op)
    {
        op(mWord, other.mWord);
        return *this;
    }
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other1, const NodeMask& other2, const WordOp& op)
    {
        op(mWord, other1.mWord, other2.mWord);
        return *this;
    }
    template<typename WordOp>
    const NodeMask& foreach(const NodeMask& other1, const NodeMask& other2, const NodeMask& other3,
                            const WordOp& op)
    {
        op(mWord, other1.mWord, other2.mWord, other3.mWord);
        return *this;
    }
    /// @brief Bitwise intersection
    const NodeMask& operator&=(const NodeMask& other)
    {
        mWord &= other.mWord;
        return *this;
    }
    /// @brief Bitwise union
    const NodeMask& operator|=(const NodeMask& other)
    {
        mWord |= other.mWord;
        return *this;
    }
    /// @brief Bitwise difference
    const NodeMask& operator-=(const NodeMask& other)
    {
        mWord &= ~other.mWord;
        return *this;
    }
    /// @brief Bitwise XOR
    const NodeMask& operator^=(const NodeMask& other)
    {
        mWord ^= other.mWord;
        return *this;
    }
    NodeMask operator!()                      const { NodeMask m(*this); m.toggle(); return m; }
    NodeMask operator&(const NodeMask& other) const { NodeMask m(*this); m &= other; return m; }
    NodeMask operator|(const NodeMask& other) const { NodeMask m(*this); m |= other; return m; }
    NodeMask operator^(const NodeMask& other) const { NodeMask m(*this); m ^= other; return m; }
    /// Return the byte size of this NodeMask
    static Index32 memUsage() { return 8; }
    /// Return the total number of on bits
    Index32 countOn() const { return CountOn(mWord); }
    ///  Return the total number of on bits
    Index32 countOff() const { return CountOff(mWord); }
    /// Set the <i>n</i>th  bit on
    void setOn(Index32 n) {
        assert( n  < 64 );
        mWord |= UINT64_C(0x01) << (n & 63);
    }
    /// Set the <i>n</i>th bit off
    void setOff(Index32 n) {
        assert( n  < 64 );
        mWord &= ~(UINT64_C(0x01) << (n & 63));
    }
    /// Set the <i>n</i>th bit to the specified state
    void set(Index32 n, bool On) { On ? this->setOn(n) : this->setOff(n); }
    /// Set all bits to the specified state
    void set(bool on) { mWord = on ? UINT64_C(0xFFFFFFFFFFFFFFFF) : UINT64_C(0x00); }
    /// Set all bits on
    void setOn() { mWord = UINT64_C(0xFFFFFFFFFFFFFFFF); }
    /// Set all bits off
    void setOff() { mWord = UINT64_C(0x00); }
    /// Toggle the state of the <i>n</i>th bit
    void toggle(Index32 n) {
        assert( n  < 64 );
        mWord ^= UINT64_C(0x01) << (n & 63);
    }
    /// Toggle the state of all bits in the mask
    void toggle() { mWord = ~mWord; }
    /// Set the first bit on
    void setFirstOn()  { this->setOn(0); }
    /// Set the last bit on
    void setLastOn()   { this->setOn(63); }
    /// Set the first bit off
    void setFirstOff() { this->setOff(0); }
    /// Set the last bit off
    void setLastOff()  { this->setOff(63); }
    /// Return true if the <i>n</i>th bit is on
    bool isOn(Index32 n) const
    {
        assert( n  < 64 );
        return 0 != (mWord & (UINT64_C(0x01) << (n & 63)));
    }
    /// Return true if the <i>n</i>th bit is off
    bool isOff(Index32 n) const {return !this->isOn(n); }
    /// Return true if all the bits are on
    bool isOn() const { return mWord == UINT64_C(0xFFFFFFFFFFFFFFFF); }
    /// Return true if all the bits are off
    bool isOff() const { return mWord == 0; }
    /// Return @c true if bits are either all off OR all on.
    /// @param isOn Takes on the values of all bits if the method
    /// returns true - else it is undefined.
    bool isConstant(bool &isOn) const
    {   isOn = this->isOn();
        return isOn || this->isOff();
    }
    Index32 findFirstOn() const { return mWord ? FindLowestOn(mWord) : 64; }
    Index32 findFirstOff() const
    {
        const Word w = ~mWord;
        return w ? FindLowestOn(w) : 64;
    }
    //@{
    /// Return the <i>n</i>th word of the bit mask, for a word of arbitrary size.
    template<typename WordT>
    WordT getWord(Index n) const
    {
        assert(n*8*sizeof(WordT) < SIZE);
        return reinterpret_cast<const WordT*>(&mWord)[n];
    }
    template<typename WordT>
    WordT& getWord(Index n)
    {
        assert(n*8*sizeof(WordT) < SIZE);
        return reinterpret_cast<WordT*>(mWord)[n];
    }
    //@}
    void save(std::ostream& os) const { os.write(reinterpret_cast<const char*>(&mWord), 8); }
    void load(std::istream& is) { is.read(reinterpret_cast<char*>(&mWord), 8); }
    void seek(std::istream& is) const { is.seekg(8, std::ios_base::cur); }
    /// @brief simple print method for debugging
    void printInfo(std::ostream& os=std::cout) const
    {
        os << "NodeMask: Dim=4, Log2Dim=2, Bit count=64, Word count=1"<<std::endl;
    }
    void printBits(std::ostream& os=std::cout) const
    {
        os << "|";
        for (Index32 i=0; i < 64; ++i) {
            if ( !(i%8) ) os << "|";
            os << this->isOn(i);
        }
        os << "||" << std::endl;
    }
    void printAll(std::ostream& os=std::cout) const
    {
        this->printInfo(os);
        this->printBits(os);
    }

    Index32 findNextOn(Index32 start) const
    {
        if (start>=64) return 64;
        const Word w = mWord & (UINT64_C(0xFFFFFFFFFFFFFFFF) << start);
        return  w ? FindLowestOn(w) : 64;
    }

    Index32 findNextOff(Index32 start) const
    {
        if (start>=64) return 64;
        const Word w = ~mWord & (UINT64_C(0xFFFFFFFFFFFFFFFF) << start);
        return  w ? FindLowestOn(w) : 64;
    }

};// NodeMask<2>


// Unlike NodeMask above this RootNodeMask has a run-time defined size.
// It is only included for backward compatibility and will likely be
// deprecated in the future!
// This class is 32-bit specefic, hence the use if Index32 vs Index!
class RootNodeMask
{
protected:
    Index32   mBitSize, mIntSize;
    Index32  *mBits;

public:
    RootNodeMask(): mBitSize(0), mIntSize(0), mBits(nullptr) {}
    RootNodeMask(Index32 bit_size):
        mBitSize(bit_size), mIntSize(((bit_size-1)>>5)+1), mBits(new Index32[mIntSize])
    {
        for (Index32 i=0; i<mIntSize; ++i) mBits[i]=0x00000000;
    }
    RootNodeMask(const RootNodeMask& B):
        mBitSize(B.mBitSize), mIntSize(B.mIntSize), mBits(new Index32[mIntSize])
    {
        for (Index32 i=0; i<mIntSize; ++i) mBits[i]=B.mBits[i];
    }
    ~RootNodeMask() {delete [] mBits;}

    void init(Index32 bit_size) {
        mBitSize = bit_size;
        mIntSize =((bit_size-1)>>5)+1;
        delete [] mBits;
        mBits = new Index32[mIntSize];
        for (Index32 i=0; i<mIntSize; ++i) mBits[i]=0x00000000;
    }

    Index getBitSize() const {return mBitSize;}

    Index getIntSize() const {return mIntSize;}

    RootNodeMask& operator=(const RootNodeMask& B) {
        if (mBitSize!=B.mBitSize) {
            mBitSize=B.mBitSize;
            mIntSize=B.mIntSize;
            delete [] mBits;
            mBits = new Index32[mIntSize];
        }
        for (Index32 i=0; i<mIntSize; ++i) mBits[i]=B.mBits[i];
        return *this;
    }

    class BaseIterator
    {
    protected:
        Index32             mPos;//bit position
        Index32             mBitSize;
        const RootNodeMask* mParent;//this iterator can't change the parent_mask!
    public:
        BaseIterator() : mPos(0), mBitSize(0), mParent(nullptr) {}
        BaseIterator(const BaseIterator&) = default;
        BaseIterator(Index32 pos, const RootNodeMask* parent):
            mPos(pos), mBitSize(parent->getBitSize()), mParent(parent) { assert(pos <= mBitSize); }
        bool operator==(const BaseIterator &iter) const {return mPos == iter.mPos;}
        bool operator!=(const BaseIterator &iter) const {return mPos != iter.mPos;}
        bool operator< (const BaseIterator &iter) const {return mPos <  iter.mPos;}
        BaseIterator& operator=(const BaseIterator& iter) {
            mPos      = iter.mPos;
            mBitSize  = iter.mBitSize;
            mParent   = iter.mParent;
            return *this;
        }

        Index32 offset() const {return mPos;}

        Index32 pos() const {return mPos;}

        bool test() const {
            assert(mPos  <= mBitSize);
            return (mPos != mBitSize);
        }

        operator bool() const {return this->test();}
    }; // class BaseIterator

    /// @note This happens to be a const-iterator!
    class OnIterator: public BaseIterator
    {
    protected:
        using BaseIterator::mPos;//bit position;
        using BaseIterator::mBitSize;//bit size;
        using BaseIterator::mParent;//this iterator can't change the parent_mask!
    public:
        OnIterator() : BaseIterator() {}
        OnIterator(Index32 pos,const RootNodeMask *parent) : BaseIterator(pos,parent) {}
        void increment() {
            assert(mParent != nullptr);
            mPos=mParent->findNextOn(mPos+1);
            assert(mPos <= mBitSize);
        }
        void increment(Index n) {
            for (Index i=0; i<n && this->next(); ++i) {}
        }
        bool next() {
            this->increment();
            return this->test();
        }
        bool operator*() const {return true;}
        OnIterator& operator++() {
            this->increment();
            return *this;
        }
    }; // class OnIterator

    class OffIterator: public BaseIterator
    {
    protected:
        using BaseIterator::mPos;//bit position;
        using BaseIterator::mBitSize;//bit size;
        using BaseIterator::mParent;//this iterator can't change the parent_mask!
    public:
        OffIterator() : BaseIterator()  {}
        OffIterator(Index32 pos,const RootNodeMask *parent) : BaseIterator(pos,parent) {}
        void increment() {
            assert(mParent != nullptr);
            mPos=mParent->findNextOff(mPos+1);
            assert(mPos <= mBitSize);
        }
        void increment(Index n) {
            for (Index i=0; i<n && this->next(); ++i) {}
        }
        bool next() {
            this->increment();
            return this->test();
        }
        bool operator*() const {return true;}
        OffIterator& operator++() {
            this->increment();
            return *this;
        }
    }; // class OffIterator

    class DenseIterator: public BaseIterator
    {
    protected:
        using BaseIterator::mPos;//bit position;
        using BaseIterator::mBitSize;//bit size;
        using BaseIterator::mParent;//this iterator can't change the parent_mask!
    public:
        DenseIterator() : BaseIterator() {}
        DenseIterator(Index32 pos,const RootNodeMask *parent) : BaseIterator(pos,parent) {}
        void increment() {
            assert(mParent != nullptr);
            mPos += 1;//carefull - the increament might go beyond the end
            assert(mPos<= mBitSize);
        }
        void increment(Index n) {
            for (Index i=0; i<n && this->next(); ++i) {}
        }
        bool next() {
            this->increment();
            return this->test();
        }
        bool operator*() const {return mParent->isOn(mPos);}
        DenseIterator& operator++() {
            this->increment();
            return *this;
        }
    }; // class DenseIterator

    OnIterator beginOn() const       { return OnIterator(this->findFirstOn(),this); }
    OnIterator endOn() const         { return OnIterator(mBitSize,this); }
    OffIterator beginOff() const     { return OffIterator(this->findFirstOff(),this); }
    OffIterator endOff() const       { return OffIterator(mBitSize,this); }
    DenseIterator beginDense() const { return DenseIterator(0,this); }
    DenseIterator endDense() const   { return DenseIterator(mBitSize,this); }

    bool operator == (const RootNodeMask &B) const {
        if (mBitSize != B.mBitSize) return false;
        for (Index32 i=0; i<mIntSize; ++i) if (mBits[i] !=  B.mBits[i]) return false;
        return true;
    }

    bool operator != (const RootNodeMask &B) const {
        if (mBitSize != B.mBitSize) return true;
        for (Index32 i=0; i<mIntSize; ++i) if (mBits[i] !=  B.mBits[i]) return true;
        return false;
    }

    //
    // Bitwise logical operations
    //
    RootNodeMask operator!() const { RootNodeMask m = *this; m.toggle(); return m; }
    const RootNodeMask& operator&=(const RootNodeMask& other) {
        assert(mIntSize == other.mIntSize);
        for (Index32 i = 0, N = std::min(mIntSize, other.mIntSize); i < N; ++i) {
            mBits[i] &= other.mBits[i];
        }
        for (Index32 i = other.mIntSize; i < mIntSize; ++i) mBits[i] = 0x00000000;
        return *this;
    }
    const RootNodeMask& operator|=(const RootNodeMask& other) {
        assert(mIntSize == other.mIntSize);
        for (Index32 i = 0, N = std::min(mIntSize, other.mIntSize); i < N; ++i) {
            mBits[i] |= other.mBits[i];
        }
        return *this;
    }
    const RootNodeMask& operator^=(const RootNodeMask& other) {
        assert(mIntSize == other.mIntSize);
        for (Index32 i = 0, N = std::min(mIntSize, other.mIntSize); i < N; ++i) {
            mBits[i] ^= other.mBits[i];
        }
        return *this;
    }
    RootNodeMask operator&(const RootNodeMask& other) const {
        RootNodeMask m(*this); m &= other; return m;
    }
    RootNodeMask operator|(const RootNodeMask& other) const {
        RootNodeMask m(*this); m |= other; return m;
    }
    RootNodeMask operator^(const RootNodeMask& other) const {
        RootNodeMask m(*this); m ^= other; return m;
    }


    Index32 getMemUsage() const {
        return static_cast<Index32>(mIntSize*sizeof(Index32) + sizeof(*this));
    }

    Index32 countOn() const {
        assert(mBits);
        Index32 n=0;
        for (Index32 i=0; i< mIntSize; ++i) n += CountOn(mBits[i]);
        return n;
    }

    Index32 countOff() const { return mBitSize-this->countOn(); }

    void setOn(Index32 i) {
        assert(mBits);
        assert( (i>>5) < mIntSize);
        mBits[i>>5] |=  1<<(i&31);
    }

    void setOff(Index32 i) {
        assert(mBits);
        assert( (i>>5) < mIntSize);
        mBits[i>>5] &=  ~(1<<(i&31));
    }

    void set(Index32 i, bool On) { On ? this->setOn(i) : this->setOff(i); }

    void setOn() {
        assert(mBits);
        for (Index32 i=0; i<mIntSize; ++i) mBits[i]=0xFFFFFFFF;
    }
    void setOff() {
        assert(mBits);
        for (Index32 i=0; i<mIntSize; ++i) mBits[i]=0x00000000;
    }
    void toggle(Index32 i) {
        assert(mBits);
        assert( (i>>5) < mIntSize);
        mBits[i>>5] ^= 1<<(i&31);
    }
    void toggle() {
        assert(mBits);
        for (Index32 i=0; i<mIntSize; ++i) mBits[i]=~mBits[i];
    }
    void setFirstOn()  { this->setOn(0); }
    void setLastOn()   { this->setOn(mBitSize-1); }
    void setFirstOff() { this->setOff(0); }
    void setLastOff()  { this->setOff(mBitSize-1); }
    bool isOn(Index32 i) const {
        assert(mBits);
        assert( (i>>5) < mIntSize);
        return ( mBits[i >> 5] & (1<<(i&31)) );
    }
    bool isOff(Index32 i) const {
        assert(mBits);
        assert( (i>>5) < mIntSize);
        return ( ~mBits[i >> 5] & (1<<(i&31)) );
    }

    bool isOn() const {
        if (!mBits) return false;//undefined is off
        for (Index32 i=0; i<mIntSize; ++i) if (mBits[i] != 0xFFFFFFFF) return false;
        return true;
    }

    bool isOff() const {
        if (!mBits) return true;//undefined is off
        for (Index32 i=0; i<mIntSize; ++i) if (mBits[i] != 0) return false;
        return true;
    }

    Index32 findFirstOn() const {
        assert(mBits);
        Index32 i=0;
        while(!mBits[i]) if (++i == mIntSize) return mBitSize;//reached end
        return 32*i + FindLowestOn(mBits[i]);
    }

    Index32 findFirstOff() const {
        assert(mBits);
        Index32 i=0;
        while(!(~mBits[i])) if (++i == mIntSize) return mBitSize;//reached end
        return 32*i + FindLowestOn(~mBits[i]);
    }

    void save(std::ostream& os) const {
        assert(mBits);
        os.write(reinterpret_cast<const char*>(mBits), mIntSize * sizeof(Index32));
    }
    void load(std::istream& is) {
        assert(mBits);
        is.read(reinterpret_cast<char*>(mBits), mIntSize * sizeof(Index32));
    }
    void seek(std::istream& is) const {
        assert(mBits);
        is.seekg(mIntSize * sizeof(Index32), std::ios_base::cur);
    }
    /// @brief simple print method for debugging
    void printInfo(std::ostream& os=std::cout) const {
        os << "RootNodeMask: Bit-size="<<mBitSize<<" Int-size="<<mIntSize<<std::endl;
    }

    void printBits(std::ostream& os=std::cout, Index32 max_out=80u) const {
        const Index32 n=(mBitSize>max_out?max_out:mBitSize);
        for (Index32 i=0; i < n; ++i) {
            if ( !(i&31) )
                os << "||";
            else if ( !(i%8) )
                os << "|";
            os << this->isOn(i);
        }
        os << "|" << std::endl;
    }

    void printAll(std::ostream& os=std::cout, Index32 max_out=80u) const {
        this->printInfo(os);
        this->printBits(os,max_out);
    }

    Index32 findNextOn(Index32 start) const {
        assert(mBits);
        Index32 n = start >> 5, m = start & 31;//initiate
        if (n>=mIntSize) return mBitSize; // check for out of bounds
        Index32 b = mBits[n];
        if (b & (1<<m)) return start;//simple case
        b &= 0xFFFFFFFF << m;// mask lower bits
        while(!b && ++n<mIntSize) b = mBits[n];// find next nonzero int
        return (!b ? mBitSize : 32*n + FindLowestOn(b));//catch last-int=0
    }

    Index32 findNextOff(Index32 start) const {
        assert(mBits);
        Index32 n = start >> 5, m = start & 31;//initiate
        if (n>=mIntSize) return mBitSize; // check for out of bounds
        Index32 b = ~mBits[n];
        if (b & (1<<m)) return start;//simple case
        b &= 0xFFFFFFFF<<m;// mask lower bits
        while(!b && ++n<mIntSize) b = ~mBits[n];// find next nonzero int
        return (!b ? mBitSize : 32*n + FindLowestOn(b));//catch last-int=0
    }

    Index32 memUsage() const {
        assert(mBits);
        return static_cast<Index32>(sizeof(Index32*)+(2+mIntSize)*sizeof(Index32));//in bytes
    }
}; // class RootNodeMask

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_NODEMASKS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
