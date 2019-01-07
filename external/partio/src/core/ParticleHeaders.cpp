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
#include "ParticleHeaders.h"
#include <map>
#include <algorithm>
#include <cassert>
#include <iostream>

using namespace Partio;

ParticleHeaders::
ParticleHeaders()
    :particleCount(0)
{
}

ParticleHeaders::
~ParticleHeaders()
{}

void ParticleHeaders::
release() const
{
    delete this;
}

int ParticleHeaders::
numParticles() const
{
    return particleCount;
}

int ParticleHeaders::
numAttributes() const
{
    return attributes.size();
}

bool ParticleHeaders::
attributeInfo(const int attributeIndex,ParticleAttribute& attribute) const
{
    if(attributeIndex<0 || attributeIndex>=(int)attributes.size()) return false;
    attribute=attributes[attributeIndex];
    return true;
}

bool ParticleHeaders::
attributeInfo(const char* attributeName,ParticleAttribute& attribute) const
{
    std::map<std::string,int>::const_iterator it=nameToAttribute.find(attributeName);
    if(it!=nameToAttribute.end()){
        attribute=attributes[it->second];
        return true;
    }
    return false;
}

void ParticleHeaders::
sort()
{
    assert(false);
}


int ParticleHeaders::
registerIndexedStr(const ParticleAttribute& attribute,const char* str)
{
    assert(false);
    return -1;
}

int ParticleHeaders::
lookupIndexedStr(const ParticleAttribute& attribute,const char* str) const
{
    assert(false);
    return -1;
}

const std::vector<std::string>& ParticleHeaders::
indexedStrs(const ParticleAttribute& attr) const
{
    static std::vector<std::string> dummy;
    assert(false);
    return dummy;
}

void ParticleHeaders::
findPoints(const float bboxMin[3],const float bboxMax[3],std::vector<ParticleIndex>& points) const
{
    assert(false);
}

float ParticleHeaders::
findNPoints(const float center[3],const int nPoints,const float maxRadius,std::vector<ParticleIndex>& points,
    std::vector<float>& pointDistancesSquared) const
{
    assert(false);
    return 0;
}

int ParticleHeaders::
findNPoints(const float center[3],int nPoints,const float maxRadius, ParticleIndex *points,
    float *pointDistancesSquared, float *finalRadius2) const
{
    assert(false);
    return 0;
}

ParticleAttribute ParticleHeaders::
addAttribute(const char* attribute,ParticleAttributeType type,const int count)
{
    // TODO: check if attribute already exists and if so what data type
    ParticleAttribute attr;
    attr.name=attribute;
    attr.type=type;
    attr.attributeIndex=attributes.size(); //  all arrays separate so we don't use this here!
    attr.count=count;
    attributes.push_back(attr);
    nameToAttribute[attribute]=attributes.size()-1;
    return attr;
}

ParticleIndex ParticleHeaders::
addParticle()
{
    ParticleIndex index=particleCount;
    particleCount++;
    return index;
}

ParticlesDataMutable::iterator ParticleHeaders::
addParticles(const int countToAdd)
{
    particleCount+=countToAdd;
    return iterator();
}

void* ParticleHeaders::
dataInternal(const ParticleAttribute& attribute,const ParticleIndex particleIndex) const
{
    assert(false);
    return 0;
}

void ParticleHeaders::
dataInternalMultiple(const ParticleAttribute& attribute,const int indexCount,
    const ParticleIndex* particleIndices,const bool sorted,char* values) const
{
    assert(false);
}

void ParticleHeaders::
dataAsFloat(const ParticleAttribute& attribute,const int indexCount,
    const ParticleIndex* particleIndices,const bool sorted,float* values) const
{
    assert(false);
}

