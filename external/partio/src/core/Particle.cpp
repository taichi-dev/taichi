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
#include "ParticleSimpleInterleave.h"
#include <iostream>
#include <string>
#include <cassert>
namespace Partio{

std::string
TypeName(ParticleAttributeType attrType)
{
    switch(attrType){
        case NONE: return "NONE";
        case VECTOR: return "VECTOR";
        case FLOAT: return "FLOAT";
        case INT: return "INT";
        case INDEXEDSTR: return "INDEXEDSTR";
        default: return 0;
    }
}

ParticlesDataMutable*
create()
{
   return new ParticlesSimple;
}

ParticlesDataMutable*
createInterleave()
{
    return new ParticlesSimpleInterleave;
}




template<ParticleAttributeType ETYPE> void
printAttr(const ParticlesData* p,const ParticleAttribute& attr,const int particleIndex)
{
    typedef typename ETYPE_TO_TYPE<ETYPE>::TYPE TYPE;
    const TYPE* data=p->data<TYPE>(attr,particleIndex);
    for(int k=0;k<attr.count;k++) std::cout<<" "<<data[k];
}

void
print(const ParticlesData* particles)
{
    std::cout<<"Particle count "<<particles->numParticles()<<std::endl;
    std::cout<<"Attribute count "<<particles->numAttributes()<<std::endl;

    std::vector<ParticleAttribute> attrs;
    for(int i=0;i<particles->numAttributes();i++){
        ParticleAttribute attr;
        particles->attributeInfo(i,attr);
        attrs.push_back(attr);
        std::cout<<"attribute "<<attr.name<<" "<<int(attr.type)<<" "<<attr.count<<std::endl;
    }

    int numToPrint=std::min(10,particles->numParticles());
    std::cout<<"num to print "<<numToPrint<<std::endl;

    ParticlesData::const_iterator it=particles->begin(),end=particles->end();
    std::vector<ParticleAccessor> accessors;
    for(size_t k=0;k<attrs.size();k++) accessors.push_back(ParticleAccessor(attrs[k]));
    for(size_t k=0;k<attrs.size();k++) it.addAccessor(accessors[k]);

    for(int i=0;i<numToPrint && it != end;i++){
        std::cout<<i<<": ";
        for(unsigned int k=0;k<attrs.size();k++){
            switch(attrs[k].type){
            case NONE:break;
            case FLOAT:
            case VECTOR:
                for(int c=0;c<attrs[k].count;c++) std::cout<<accessors[k].raw<float>(it)[c];
                break;
            case INT:
                for(int c=0;c<attrs[k].count;c++) std::cout<<accessors[k].raw<int>(it)[c];
                break;
            case INDEXEDSTR:
                for(int c=0;c<attrs[k].count;c++) std::cout<<accessors[k].raw<int>(it)[c];
                break;
            }
        }
        std::cout<<std::endl;
    }
}


}
