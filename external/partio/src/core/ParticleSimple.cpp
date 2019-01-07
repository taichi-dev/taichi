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

#ifdef PARTIO_WIN32
#    define NOMINMAX
#endif

#include "ParticleSimple.h"
#include "ParticleCaching.h"
#include <map>
#include <algorithm>
#include <cassert>
#include <iostream>

#include "KdTree.h"


using namespace Partio;

ParticlesSimple::
ParticlesSimple()
    :particleCount(0),allocatedCount(0),kdtree(0)
{
}

ParticlesSimple::
~ParticlesSimple()
{
    for(unsigned int i=0;i<attributeData.size();i++) free(attributeData[i]);
    delete kdtree;
}

void ParticlesSimple::
release() const
{
    freeCached(const_cast<ParticlesSimple*>(this));
}

int ParticlesSimple::
numParticles() const
{
    return particleCount;
}

int ParticlesSimple::
numAttributes() const
{
    return attributes.size();
}

bool ParticlesSimple::
attributeInfo(const int attributeIndex,ParticleAttribute& attribute) const
{
    if(attributeIndex<0 || attributeIndex>=(int)attributes.size()) return false;
    attribute=attributes[attributeIndex];
    return true;
}

bool ParticlesSimple::
attributeInfo(const char* attributeName,ParticleAttribute& attribute) const
{
    std::map<std::string,int>::const_iterator it=nameToAttribute.find(attributeName);
    if(it!=nameToAttribute.end()){
        attribute=attributes[it->second];
        return true;
    }
    return false;
}


void ParticlesSimple::
sort()
{
    ParticleAttribute attr;
    bool foundPosition=attributeInfo("position",attr);
    if(!foundPosition){
        std::cerr<<"Partio: sort, Failed to find position in particle"<<std::endl;
        return;
    }else if(attr.type!=VECTOR || attr.count!=3){
        std::cerr<<"Partio: sort, position attribute is not a vector of size 3"<<std::endl;
        return;
    }

    const ParticleIndex baseParticleIndex=0;
    const float* data=this->data<float>(attr,baseParticleIndex); // contiguous assumption used here
    KdTree<3>* kdtree_temp=new KdTree<3>();
    kdtree_temp->setPoints(data,numParticles());
    kdtree_temp->sort();

    kdtree_mutex.lock();
    // TODO: this is not threadsafe!
    if(kdtree) delete kdtree;
    kdtree=kdtree_temp;
    kdtree_mutex.unlock();
}

void ParticlesSimple::
findPoints(const float bboxMin[3],const float bboxMax[3],std::vector<ParticleIndex>& points) const
{
    if(!kdtree){
        std::cerr<<"Partio: findPoints without first calling sort()"<<std::endl;
        return;
    }

    BBox<3> box(bboxMin);box.grow(bboxMax);

    int startIndex=points.size();
    kdtree->findPoints(points,box);
    // remap points found in findPoints to original index space
    for(unsigned int i=startIndex;i<points.size();i++){
        points[i]=kdtree->id(points[i]);
    }
}

float ParticlesSimple::
findNPoints(const float center[3],const int nPoints,const float maxRadius,std::vector<ParticleIndex>& points,
    std::vector<float>& pointDistancesSquared) const
{
    if(!kdtree){
        std::cerr<<"Partio: findNPoints without first calling sort()"<<std::endl;
        return 0;
    }

    //assert(sizeof(ParticleIndex)==sizeof(uint64_t));
    //std::vector<uint64_t>& rawPoints=points;
    float maxDistance=kdtree->findNPoints(points,pointDistancesSquared,center,nPoints,maxRadius);
    // remap all points since findNPoints clears array
    for(unsigned int i=0;i<points.size();i++){
        ParticleIndex index=kdtree->id(points[i]);
        points[i]=index;
    }
    return maxDistance;
}

int ParticlesSimple::
findNPoints(const float center[3],int nPoints,const float maxRadius, ParticleIndex *points,
    float *pointDistancesSquared, float *finalRadius2) const
{
    if(!kdtree){
        std::cerr<<"Partio: findNPoints without first calling sort()"<<std::endl;
        return 0;
    }

    int count = kdtree->findNPoints (points, pointDistancesSquared, finalRadius2, center, nPoints, maxRadius);
    // remap all points since findNPoints clears array
    for(int i=0; i < count; i++){
        ParticleIndex index = kdtree->id(points[i]);
        points[i]=index;
    }
    return count;
}

ParticleAttribute ParticlesSimple::
addAttribute(const char* attribute,ParticleAttributeType type,const int count)
{
    if(nameToAttribute.find(attribute) != nameToAttribute.end()){
        std::cerr<<"Partio: addAttribute failed because attr '"<<attribute<<"'"<<" already exists"<<std::endl;
        return ParticleAttribute();
    }
    // TODO: check if attribute already exists and if so what data type
    ParticleAttribute attr;
    attr.name=attribute;
    attr.type=type;
    attr.attributeIndex=attributes.size(); //  all arrays separate so we don't use this here!
    attr.count=count;
    attributes.push_back(attr);
    nameToAttribute[attribute]=attributes.size()-1;

    int stride=TypeSize(type)*count;
    attributeStrides.push_back(stride);
    char* dataPointer=(char*)malloc(allocatedCount*stride);
    attributeData.push_back(dataPointer);
    attributeOffsets.push_back(dataPointer-(char*)0);
    attributeIndexedStrs.push_back(IndexedStrTable());

    return attr;
}

ParticleIndex ParticlesSimple::
addParticle()
{
    if(allocatedCount==particleCount){
        allocatedCount=std::max(10,std::max(allocatedCount*3/2,particleCount));
        for(unsigned int i=0;i<attributes.size();i++)
            attributeData[i]=(char*)realloc(attributeData[i],(size_t)attributeStrides[i]*(size_t)allocatedCount);
    }
    ParticleIndex index=particleCount;
    particleCount++;
    return index;
}

ParticlesDataMutable::iterator ParticlesSimple::
addParticles(const int countToAdd)
{
    if(particleCount+countToAdd>allocatedCount){
        // TODO: this should follow 2/3 rule
        allocatedCount=allocatedCount+countToAdd;
        for(unsigned int i=0;i<attributes.size();i++){
            attributeData[i]=(char*)realloc(attributeData[i],(size_t)attributeStrides[i]*(size_t)allocatedCount);
            attributeOffsets[i]=attributeData[i]-(char*)0;
        }
    }
    //int offset=particleCount;
    particleCount+=countToAdd;
    iterator it=setupIterator();
    // TODO: this needs to advance the iterator to the appropriate spot
    return it;
}

ParticlesDataMutable::iterator ParticlesSimple::
setupIterator()
{
    if(numParticles()==0) return ParticlesDataMutable::iterator();
    return ParticlesDataMutable::iterator(this,0,numParticles()-1);
}

ParticlesData::const_iterator ParticlesSimple::
setupConstIterator() const
{
    if(numParticles()==0) return ParticlesDataMutable::const_iterator();
    return ParticlesData::const_iterator(this,0,numParticles()-1);
}

void ParticlesSimple::
setupIteratorNextBlock(Partio::ParticleIterator<false>& iterator)
{
    iterator=end();
}

void ParticlesSimple::
setupIteratorNextBlock(Partio::ParticleIterator<true>& iterator) const
{
    iterator=ParticlesData::end();
}


void ParticlesSimple::
setupAccessor(Partio::ParticleIterator<false>& iterator,ParticleAccessor& accessor)
{
    accessor.stride=accessor.count*sizeof(float);
    accessor.basePointer=attributeData[accessor.attributeIndex];
}

void ParticlesSimple::
setupAccessor(Partio::ParticleIterator<true>& iterator,ParticleAccessor& accessor) const
{
    accessor.stride=accessor.count*sizeof(float);
    accessor.basePointer=attributeData[accessor.attributeIndex];
}

void* ParticlesSimple::
dataInternal(const ParticleAttribute& attribute,const ParticleIndex particleIndex) const
{
    assert(attribute.attributeIndex>=0 && attribute.attributeIndex<(int)attributes.size());
    return attributeData[attribute.attributeIndex]+attributeStrides[attribute.attributeIndex]*particleIndex;
}

void ParticlesSimple::
dataInternalMultiple(const ParticleAttribute& attribute,const int indexCount,
    const ParticleIndex* particleIndices,const bool sorted,char* values) const
{
    assert(attribute.attributeIndex>=0 && attribute.attributeIndex<(int)attributes.size());

    char* base=attributeData[attribute.attributeIndex];
    int bytes=attributeStrides[attribute.attributeIndex];
    for(int i=0;i<indexCount;i++)
        memcpy(values+bytes*i,base+particleIndices[i]*bytes,bytes);
}

void ParticlesSimple::
dataAsFloat(const ParticleAttribute& attribute,const int indexCount,
    const ParticleIndex* particleIndices,const bool sorted,float* values) const
{
    assert(attribute.attributeIndex>=0 && attribute.attributeIndex<(int)attributes.size());

    if(attribute.type==FLOAT || attribute.type==VECTOR) dataInternalMultiple(attribute,indexCount,particleIndices,sorted,(char*)values);
    else if(attribute.type==INT || attribute.type==INDEXEDSTR){
        char* attrrawbase=attributeData[attribute.attributeIndex];
        int* attrbase=(int*)attrrawbase;
        int count=attribute.count;
        for(int i=0;i<indexCount;i++) for(int k=0;k<count;k++) values[i*count+k]=(int)attrbase[particleIndices[i]*count+k];
    }
}

int ParticlesSimple::
registerIndexedStr(const ParticleAttribute& attribute,const char* str)
{
    IndexedStrTable& table=attributeIndexedStrs[attribute.attributeIndex];
    std::map<std::string,int>::const_iterator it=table.stringToIndex.find(str);
    if(it!=table.stringToIndex.end()) return it->second;
    int newIndex=table.strings.size();
    table.strings.push_back(str);
    table.stringToIndex[str]=newIndex;
    return newIndex;
}

int ParticlesSimple::
lookupIndexedStr(Partio::ParticleAttribute const &attribute, char const *str) const
{
    const IndexedStrTable& table=attributeIndexedStrs[attribute.attributeIndex];
    std::map<std::string,int>::const_iterator it=table.stringToIndex.find(str);
    if(it!=table.stringToIndex.end()) return it->second;
    return -1;
}

const std::vector<std::string>& ParticlesSimple::
indexedStrs(const ParticleAttribute& attr) const
{
    const IndexedStrTable& table=attributeIndexedStrs[attr.attributeIndex];
    return table.strings;
}

