/*
PARTIO SOFTWARE
Copyright 2010 Disney Enterprises, Inc. All rights reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.

* The names "Disney", "Walt Disney Pictures", "Walt Disney Animation
Studios" or the names of its contributors may NOT be used to
endorse or promote products derived from this software without
specific prior written permission from Walt Disney Pictures.

Disclaimer: THIS SOFTWARE IS PROVIDED BY WALT DISNEY PICTURES AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, NONINFRINGEMENT AND TITLE ARE DISCLAIMED.
IN NO EVENT SHALL WALT DISNEY PICTURES, THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND BASED ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
*/
#ifndef _PartioParticleIterator_h_
#define _PartioParticleIterator_h_

#include <cassert>
#include <vector>
#include <iostream>
#include "PartioAttribute.h"

namespace Partio{

class  ParticlesData;
struct ParticleAccessor;

//! Data
/*!
  This class represents a piece of data stored in a particle attribute.
  The only allowed values are float and d
*/
template<class T,int d>
struct Data
{
    T x[d];

    const T& operator[](const int i) const {return x[i];}
    T& operator[](const int i) {return x[i];}
};
typedef Data<int,1> DataI;
typedef Data<float,1> DataF;
typedef Data<float,3> DataV;


template<bool constant> class ParticleIterator;

struct Provider
{
    virtual void setupIteratorNextBlock(ParticleIterator<true>& iterator) const=0;
    virtual void setupIteratorNextBlock(ParticleIterator<false>& iterator)=0;
    virtual void setupAccessor(ParticleIterator<true>& iterator,ParticleAccessor& accessor) const=0;
    virtual void setupAccessor(ParticleIterator<false>& iterator,ParticleAccessor& accessor)=0;
    virtual ~Provider(){}
};

template<bool constant>
struct PROVIDER
{
    typedef Provider TYPE;
};
template<>
struct PROVIDER<true>
{
    typedef const Provider TYPE;
};

// TODO: non copyable
struct ParticleAccessor
{
    int stride;
    char* basePointer;
    int attributeIndex; // index of attribute opaque, do not touch
    int count;
private:
    ParticleAttributeType type;
    
    ParticleAccessor* next;

public:
    ParticleAccessor(const ParticleAttribute& attr)
        :stride(0),basePointer(0),attributeIndex(attr.attributeIndex),
        count(attr.count),type(attr.type),next(0)
    {}

    template<class TDATA,class TITERATOR> TDATA* raw(const TITERATOR& it)
    {return reinterpret_cast<TDATA*>(basePointer+it.index*stride);}

    template<class TDATA,class TITERATOR> const TDATA* raw(const TITERATOR& it) const
    {return reinterpret_cast<const TDATA*>(basePointer+it.index*stride);}

    template<class TDATA,class TITERATOR> TDATA& data(const TITERATOR& it)
    {return *reinterpret_cast<TDATA*>(basePointer+it.index*stride);}

    template<class TDATA,class TITERATOR> const TDATA& data(const TITERATOR& it) const
    {return *reinterpret_cast<const TDATA*>(basePointer+it.index*stride);}

    friend class ParticleIterator<true>;
    friend class ParticleIterator<false>;
};


template<bool constant=false>
class ParticleIterator
{
public:
private:
    typedef typename PROVIDER<constant>::TYPE PROVIDER;

    //! Delegate, null if the iterator is false
    PROVIDER* particles;

public:
    //! Start of non-interleaved index of contiguous block
    size_t index;
private:

    //! End of non-interleaved index of contiguous block
    size_t indexEnd;

    //! This is used for both non-interleaved and interleaved particle attributes
    ParticleAccessor* accessors;

public:
    //! Construct an invalid iterator
    ParticleIterator()
        :particles(0),index(0),indexEnd(0),accessors(0)
    {}

    //! Copy constructor. NOTE: Invalidates any accessors that have been registered with it
    ParticleIterator(const ParticleIterator& other)
		:particles(other.particles),index(other.index),indexEnd(other.indexEnd),accessors(0)
    {}

    //! Construct an iterator with iteration parameters. This is typically only
    //! called by implementations of Particle (not by users). For users, use
    //! begin() and end() on the particle type
    ParticleIterator(PROVIDER* particles,size_t index,size_t indexEnd)
        :particles(particles),index(index),indexEnd(indexEnd)
    {}

    //! Whether the iterator is valid
    bool valid() const
    {return particles;}

    //! Increment the iterator (postfix). Prefer the prefix form below to this one.
    ParticleIterator operator++(int)
    {
        ParticleIterator newIt(*this);
        index++;
        return newIt;
    }

    //! Increment the iterator (prefix).
    ParticleIterator& operator++()
    {
        index++;
        // TODO: make particles==0 check unnecessary by using indexEnd=0 to signify invalid iterator
        if((index>indexEnd) && particles) particles->setupIteratorNextBlock(*this);
        return *this;
    }

    //! Iterator comparison equals
    bool operator==(const ParticleIterator& other)
    {
        // TODO: this is really really expensive
        // TODO: this needs a block or somethingt o say which segment it is
        return particles==other.particles && index==other.index;
    }
     
    //! Iterator comparison not-equals
    bool operator!=(const ParticleIterator& other)
    {
        if(other.particles!=particles) return true; // if not same delegate
        else if(particles==0) return false; // if both are invalid iterators
        else return !(*this==other);
    }
    
    void addAccessor(ParticleAccessor& newAccessor)
    {
        newAccessor.next=accessors;
        accessors=&newAccessor;
        if(particles) particles->setupAccessor(*this,newAccessor);
    }


    // TODO: add copy constructor that wipes out accessor linked list

};

template<class T,int d>
std::ostream& operator<<(std::ostream& output,const Data<T,d>& v)
{
    output<<v[0];
    for(int i=1;i<d;i++) output<< " " << v[i];
    return output;
}


}

#endif
