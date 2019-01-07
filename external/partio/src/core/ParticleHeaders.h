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
#ifndef _ParticlesHeaders_h_
#define _ParticlesHeaders_h_

#include "../Partio.h"
namespace Partio{

class ParticleHeaders:public ParticlesDataMutable
{
public:
    ParticleHeaders();
    void release() const;
protected:
    virtual ~ParticleHeaders();

    int numAttributes() const;
    int numParticles() const;
    bool attributeInfo(const char* attributeName,ParticleAttribute& attribute) const;
    bool attributeInfo(const int attributeInfo,ParticleAttribute& attribute) const;

    int registerIndexedStr(const ParticleAttribute& attribute,const char* str);
    int lookupIndexedStr(const ParticleAttribute& attribute,const char* str) const;
    const std::vector<std::string>& indexedStrs(const ParticleAttribute& attr) const;

    virtual void dataAsFloat(const ParticleAttribute& attribute,const int indexCount,
        const ParticleIndex* particleIndices,const bool sorted,float* values) const;

    void sort();

    void findPoints(const float bboxMin[3],const float bboxMax[3],std::vector<ParticleIndex>& points) const;
    float findNPoints(const float center[3],int nPoints,const float maxRadius,
        std::vector<ParticleIndex>& points,std::vector<float>& pointDistancesSquared) const;
    int findNPoints(const float center[3],int nPoints,const float maxRadius,
        ParticleIndex *points, float *pointDistancesSquared, float *finalRadius2) const;

    ParticleAttribute addAttribute(const char* attribute,ParticleAttributeType type,const int count);
    ParticleIndex addParticle();
    iterator addParticles(const int count);

    const_iterator setupConstIterator() const
    {return const_iterator();}

    iterator setupIterator()
    {return iterator();}

private:
    void* dataInternal(const ParticleAttribute& attribute,const ParticleIndex particleIndex) const;
    void dataInternalMultiple(const ParticleAttribute& attribute,const int indexCount,
        const ParticleIndex* particleIndices,const bool sorted,char* values) const;

private:
    int particleCount;
    std::vector<ParticleAttribute> attributes;
    std::map<std::string,int> nameToAttribute;

};
}
#endif
