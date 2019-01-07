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

/*!
  The interface of the particle API (Partio)
  what type the primitive is, how many instances of the primitive there, name of
  the attribute and an index which speeds lookups of data
*/
#ifndef _Partioh_
#define _Partioh_

#include <string>
#include <vector>
#include <map>
#include <stdint.h>
#include "PartioAttribute.h"
#include "PartioIterator.h"

namespace Partio{

//! Opaque random access method to a single particle. No number is implied or guaranteed.
typedef uint64_t ParticleIndex;

class ParticlesData;
// Particle Collection Interface
//!  Particle Collection Interface
/*!
  This class provides ways of accessing basic information about particles,
  the number in the set, the attribute names and types, etc. No actual
  data can be read or written.
*/
class ParticlesInfo
{
protected:
    virtual ~ParticlesInfo() {}
public:
    friend void freeCached(ParticlesData* particles);

    //! Frees the memory if this particle set was created with create() or release()
    //! Reduces reference count if it was obtained with readCached()
    //! and if the ref count hits zero, frees the memory
    virtual void release() const=0;

    //! Number of particles in the structure.
    virtual int numAttributes() const=0;

    //! Number of per-particle attributes.
    virtual int numParticles() const=0;

    //! Lookup an attribute by name and store a handle to the attribute.
    virtual bool attributeInfo(const char* attributeName,ParticleAttribute& attribute) const=0;

    //! Lookup an attribute by index and store a handle to the attribute.
    virtual bool attributeInfo(const int attributeInfo,ParticleAttribute& attribute) const=0;
};

// Particle Data Interface
//!  Particle Data Interface
/*!
  This interface provides the ability to read data attributes for given particles
  and search for nearest neighbors   using KD-Trees.
*/
class ParticlesData:public ParticlesInfo
{
protected:
    virtual ~ParticlesData() {}
public:

    typedef ParticleIterator<true> const_iterator;

    //! Fill the user supplied values array with data corresponding to the given
    //! list of particles. Specify whether or not your indices are sorted.
    //! note if T is void, then type checking is disabled.
    template<class T> inline void data(const ParticleAttribute& attribute,
        const int indexCount,const ParticleIndex* particleIndices,const bool sorted,T* values)
    {
    	assert(typeCheck<T>(attribute.type));
        dataInternalMultiple(attribute,indexCount,particleIndices,sorted,(char*)values);
    }

    template<class T> inline const T* data(const ParticleAttribute& attribute,
        const ParticleIndex particleIndex) const
    {
        // TODO: add type checking
        return static_cast<T*>(dataInternal(attribute,particleIndex));
    }

    /// All indexed strings for an attribute
    virtual const std::vector<std::string>& indexedStrs(const ParticleAttribute& attr) const=0;

    /// Looks up the index for a given string for a given attribute, returns -1 if not found
    virtual int lookupIndexedStr(const ParticleAttribute& attribute,const char* str) const=0;

    //! Fill the user supplied values array with data corresponding to the given
    //! list of particles. Specify whether or not your indices are sorted. Attributes
    //! that are not floating types are automatically casted before being placed
    //! in values.
    virtual void dataAsFloat(const ParticleAttribute& attribute,const int indexCount,
        const ParticleIndex* particleIndices,const bool sorted,float* values) const=0;

    //! Find the points within the bounding box specified.
    //! Must call sort() before using this function
    //! NOTE: points array is not pre-cleared.
    virtual void findPoints(const float bboxMin[3],const float bboxMax[3],
        std::vector<ParticleIndex>& points) const=0;

    //! Find the N nearest neighbors that are within maxRadius distance using STL types
    //! (measured in standard 2-norm). If less than N are found within the
    //! radius, the search radius is not increased.
    //! NOTE: points/pointsDistancesSquared are cleared before use.
    //! Must call sort() before using this function
    virtual float findNPoints(const float center[3],int nPoints,const float maxRadius,
        std::vector<ParticleIndex>& points,std::vector<float>& pointDistancesSquared) const=0;

    //! Find the N nearest neighbors that are within maxRadius distance using POD types
    //! NOTE: returns the number of found points and leaves in finalRadius2 the
    //! square of the final search radius used
    virtual int findNPoints(const float center[3],int nPoints,const float maxRadius,
        ParticleIndex *points, float *pointDistancesSquared, float *finalRadius2) const=0;

    //! Produce a const iterator
    virtual const_iterator setupConstIterator() const=0;

    //! Produce a beginning iterator for the particles
    const_iterator begin() const
    {return setupConstIterator();}

    //! Produce a ending iterator for the particles
    const_iterator end() const
    {return const_iterator();}

private:
    virtual void* dataInternal(const ParticleAttribute& attribute,const ParticleIndex particleIndex) const=0;
    virtual void dataInternalMultiple(const ParticleAttribute& attribute,const int indexCount,
        const ParticleIndex* particleIndices,const bool sorted,char* values) const=0;
};

// Particle Mutable Data Interface
//!  Particle Mutable Data Interface
/*!
  This interface provides the ability to write data attributes, add attributes,
  add particles, etc.
*/
class ParticlesDataMutable:public ParticlesData
{
protected:
    virtual ~ParticlesDataMutable(){}

public:

    typedef ParticleIterator<false> iterator;

    //! Get a pointer to the data corresponding to the given particleIndex and
    //! attribute given by the attribute handle.
    template<class T> inline T* dataWrite(const ParticleAttribute& attribute,
        const ParticleIndex particleIndex) const
    {
        // TODO: add type checking
        return static_cast<T*>(dataInternal(attribute,particleIndex));
    }

    /// Returns a token for the given string. This allows efficient storage of string data
    virtual int registerIndexedStr(const ParticleAttribute& attribute,const char* str)=0;

    //! Preprocess the data for finding nearest neighbors by sorting into a
    //! KD-Tree. Note: all particle pointers are invalid after this call.
    virtual void sort()=0;

    //! Adds an attribute to the particle with the provided name, type and count
    virtual ParticleAttribute addAttribute(const char* attribute,ParticleAttributeType type,
        const int count)=0;

    //! Add a particle to the particle set. Returns the offset to the particle
    virtual ParticleIndex addParticle()=0;

    //! Add a set of particles to the particle set. Returns the offset to the
    //! first particle
    virtual iterator addParticles(const int count)=0;

    //! Produce a beginning iterator for the particles
    iterator begin()
    {return setupIterator();}

    //! Produce a ending iterator for the particles
    iterator end()
    {return iterator();}

    //! Produce a const iterator
    virtual iterator setupIterator()=0;

private:
    virtual void* dataInternal(const ParticleAttribute& attribute,const ParticleIndex particleIndex) const=0;
};

//! Provides an empty particle instance, freed with p->release()
ParticlesDataMutable* create();

ParticlesDataMutable* createInterleave();

//! Provides read/write access to a particle set stored in a file
//! freed with p->release()
ParticlesDataMutable* read(const char* filename);

//! Provides read access to a particle headers (number of particles
//! and attribute information, much cheapeer
ParticlesInfo* readHeaders(const char* filename);

//! Provides access to a particle set stored in a file
//! if filename ends with .gz or forceCompressed is true, the file is compressed.
void write(const char* filename,const ParticlesData&,const bool forceCompressed=false);


//! Cached (only one copy) read only way to read a particle file
/*!
  Loads a file read-only if not already in memory, otherwise returns
  already loaded item. Pointer is owned by Partio and must be releasedwith
  p->release(); (will not be deleted if others are also holding).
  If you want to do finding neighbors give true to sort
*/
ParticlesData* readCached(const char* filename,const bool sort);

//! Begin accessing data in a cached file
/*!
  Indicates to Partio that data access from a cached particle set will
  start. The sent in particles pointer must be from a readCached()
  call, not from read() or create(). Attributes can be read before this call.
*/
void beginCachedAccess(ParticlesData* particles);

//! End accessing data in a cached file
/*!
  Indicates to Partio that data from a cached particle read will
  end. The sent in particles pointer must be from a readCached()
  call, not from read() or create(). This allows the particle API
  to free all data pages, if they are needed.
*/
void endCachedAccess(ParticlesData* particles);

//! Prints a subset of particle data in a textual form
void print(const ParticlesData* particles);

}
#endif
