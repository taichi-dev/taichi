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

#include "ParticleSimpleInterleave.h"
#include "ParticleCaching.h"
#include <map>
#include <algorithm>
#include <cassert>
#include <iostream>

#include "KdTree.h"


using namespace Partio;

ParticlesSimpleInterleave::
ParticlesSimpleInterleave()
    :particleCount(0),allocatedCount(0),data(0),stride(0),kdtree(0)
{
}

ParticlesSimpleInterleave::
~ParticlesSimpleInterleave()
{
    free(data);
    delete kdtree;
}

void ParticlesSimpleInterleave::
release() const
{
    freeCached(const_cast<ParticlesSimpleInterleave*>(this));
}


int ParticlesSimpleInterleave::
numParticles() const
{
    return particleCount;
}

int ParticlesSimpleInterleave::
numAttributes() const
{
    return attributes.size();
}


bool ParticlesSimpleInterleave::
attributeInfo(const int attributeIndex,ParticleAttribute& attribute) const
{
    if(attributeIndex<0 || attributeIndex>=(int)attributes.size()) return false;
    attribute=attributes[attributeIndex];
    return true;
}

bool ParticlesSimpleInterleave::
attributeInfo(const char* attributeName,ParticleAttribute& attribute) const
{
    std::map<std::string,int>::const_iterator it=nameToAttribute.find(attributeName);
    if(it!=nameToAttribute.end()){
        attribute=attributes[it->second];
        return true;
    }
    return false;
}

void ParticlesSimpleInterleave::
sort()
{
#if 0
    ParticleAttribute attr;
    bool foundPosition=attributeInfo("position",attr);
    if(!foundPosition){
        std::cerr<<"Partio: sort, Failed to find position in particle"<<std::endl;
        return;
    }else if(attr.type!=VECTOR || attr.count!=3){
        std::cerr<<"Partio: sort, position attribute is not a vector of size 3"<<std::endl;
        return;
    }

    const float* data=this->data<float>(attr,0); // contiguous assumption used here
    KdTree<3>* kdtree_temp=new KdTree<3>();
    kdtree_temp->setPoints(data,numParticles());
    kdtree_temp->sort();

    kdtree_mutex.lock();
    // TODO: this is not threadsafe!
    if(kdtree) delete kdtree;
    kdtree=kdtree_temp;
    kdtree_mutex.unlock();
#endif
}

void ParticlesSimpleInterleave::
findPoints(const float bboxMin[3],const float bboxMax[3],std::vector<ParticleIndex>& points) const
{
#if 0
    if(!kdtree){
        std::cerr<<"Partio: findPoints without first calling sort()"<<std::endl;
        return;
    }

    BBox<3> box(bboxMin);box.grow(bboxMax);

    int startIndex=points.size();
    kdtree->findPoints(points,box);
    // remap points found in findPoints to original index spac
    for(unsigned int i=startIndex;i<points.size();i++) points[i]=kdtree->id(points[i]);
#endif
}

float ParticlesSimpleInterleave::
findNPoints(const float center[3],const int nPoints,const float maxRadius,std::vector<ParticleIndex>& points,
    std::vector<float>& pointDistancesSquared) const
{
#if 0
    if(!kdtree){
        std::cerr<<"Partio: findNPoints without first calling sort()"<<std::endl;
        return 0;
    }

    float maxDistance=kdtree->findNPoints(points,pointDistancesSquared,center,nPoints,maxRadius);
    // remap all points since findNPoints clears array
    for(unsigned int i=0;i<points.size();i++) points[i]=kdtree->id(points[i]);
    return maxDistance;
#endif
    return 0;
}

int ParticlesSimpleInterleave::
findNPoints(const float center[3],int nPoints,const float maxRadius, ParticleIndex *points,
    float *pointDistancesSquared, float *finalRadius2) const
{
    // TODO: I guess they don't support this lookup here
    return 0;
}


ParticleAttribute ParticlesSimpleInterleave::
addAttribute(const char* attribute,ParticleAttributeType type,const int count)
{
	//std::cerr<< "AddAttribute interleave" << std::endl;
    if(nameToAttribute.find(attribute) != nameToAttribute.end()){
        std::cerr<<"Partio: addAttribute failed because attr '"<<attribute<<"'"<<" already exists"<<std::endl;
        return ParticleAttribute();
    }
    ParticleAttribute attr;
    attr.name=attribute;
    attr.type=type;
    attr.attributeIndex=attributes.size(); //  all arrays separate so we don't use this here!
    attr.count=count;
    attributes.push_back(attr);
    nameToAttribute[attribute]=attributes.size()-1;

    // repackage data for new attribute
    int oldStride=stride;
    int newStride=stride+TypeSize(type)*count;
    char* newData=(char*)malloc((size_t)allocatedCount*(size_t)newStride);
    if(data){
        char* ptrNew=newData;
        char* ptrOld=data;
        for(int i=0;i<particleCount;i++){
            memcpy(ptrNew,ptrOld,oldStride);
            ptrNew+=newStride;
            ptrOld+=oldStride;
        }
    }
    free(data);
    data=newData;
    stride=newStride;
    attributeOffsets.push_back(oldStride);
	attributeIndexedStrs.push_back(IndexedStrTable());

    return attr;
}

ParticleIndex ParticlesSimpleInterleave::
addParticle()
{
    if(allocatedCount==particleCount){
        allocatedCount=std::max(10,std::max(allocatedCount*3/2,particleCount));
        data=(char*)realloc(data,(size_t)stride*(size_t)allocatedCount);
    }
    return particleCount++;
}

ParticlesDataMutable::iterator ParticlesSimpleInterleave::
addParticles(const int countToAdd)
{
    if(particleCount+countToAdd>allocatedCount){
        while(allocatedCount<particleCount+countToAdd)
            allocatedCount=std::max(10,std::max(allocatedCount*3/2,particleCount));
        data=(char*)realloc(data,(size_t)stride*(size_t)allocatedCount);
    }
    // int offset=particleCount;
    particleCount+=countToAdd;
    // TODO: make this return the right stuff
    return begin();
    //return offset;
}

ParticlesDataMutable::iterator ParticlesSimpleInterleave::
setupIterator()
{
    if(numParticles()==0) return ParticlesDataMutable::iterator();
    return ParticlesDataMutable::iterator(this,0,numParticles()-1);
}

ParticlesData::const_iterator ParticlesSimpleInterleave::
setupConstIterator() const
{
    if(numParticles()==0) return ParticlesDataMutable::const_iterator();
    return ParticlesData::const_iterator(this,0,numParticles()-1);
}

void ParticlesSimpleInterleave::
setupIteratorNextBlock(Partio::ParticleIterator<false>& iterator)
{
    iterator=end();
}

void ParticlesSimpleInterleave::
setupIteratorNextBlock(Partio::ParticleIterator<true>& iterator) const
{
    iterator=ParticlesData::end();
}

void ParticlesSimpleInterleave::
setupAccessor(Partio::ParticleIterator<false>& iterator,ParticleAccessor& accessor)
{
    accessor.stride=stride;
    accessor.basePointer=data+attributeOffsets[accessor.attributeIndex];
}

void ParticlesSimpleInterleave::
setupAccessor(Partio::ParticleIterator<true>& iterator,ParticleAccessor& accessor) const
{
    accessor.stride=stride;
    accessor.basePointer=data+attributeOffsets[accessor.attributeIndex];
}


void* ParticlesSimpleInterleave::
dataInternal(const ParticleAttribute& attribute,const ParticleIndex particleIndex) const
{
    return data+particleIndex*stride+attributeOffsets[attribute.attributeIndex];
}

void ParticlesSimpleInterleave::
dataInternalMultiple(const ParticleAttribute& attribute,const int indexCount,
    const ParticleIndex* particleIndices,const bool sorted,char* values) const
{
#if 0
    assert(attribute.attributeIndex>=0 && attribute.attributeIndex<(int)attributes.size());

    char* base=attributeData[attribute.attributeIndex];
    int bytes=attributeStrides[attribute.attributeIndex];
    for(int i=0;i<indexCount;i++)
        memcpy(values+bytes*i,base+particleIndices[i]*bytes,bytes);
#endif
}

void ParticlesSimpleInterleave::
dataAsFloat(const ParticleAttribute& attribute,const int indexCount,
    const ParticleIndex* particleIndices,const bool sorted,float* values) const
{
#if 0
    assert(attribute.attributeIndex>=0 && attribute.attributeIndex<(int)attributes.size());

    if(attribute.type==FLOAT || attribute.type==VECTOR) dataInternalMultiple(attribute,indexCount,particleIndices,sorted,(char*)values);
    else if(attribute.type==INT){
        char* attrrawbase=attributeData[attribute.attributeIndex];
        int* attrbase=(int*)attrrawbase;
        int count=attribute.count;
        for(int i=0;i<indexCount;i++) for(int k=0;k<count;k++) values[i*count+k]=(int)attrbase[particleIndices[i]*count+k];
    }
#endif
}


int ParticlesSimpleInterleave::
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

int ParticlesSimpleInterleave::
lookupIndexedStr(Partio::ParticleAttribute const &attribute, char const *str) const
{
    const IndexedStrTable& table=attributeIndexedStrs[attribute.attributeIndex];
    std::map<std::string,int>::const_iterator it=table.stringToIndex.find(str);
    if(it!=table.stringToIndex.end()) return it->second;
    return -1;
}

const std::vector<std::string>& ParticlesSimpleInterleave::
indexedStrs(const ParticleAttribute& attr) const
{
    const IndexedStrTable& table=attributeIndexedStrs[attr.attributeIndex];
    return table.strings;
}

