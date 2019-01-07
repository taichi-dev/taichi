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

#ifndef _PartioParticleAttribute_h_
#define _PartioParticleAttribute_h_
namespace Partio{

// Particle Types
enum ParticleAttributeType {NONE=0,VECTOR=1,FLOAT=2,INT=3,INDEXEDSTR=4};

template<ParticleAttributeType ENUMTYPE> struct ETYPE_TO_TYPE
{struct UNUSABLE;typedef UNUSABLE TYPE;};
template<> struct ETYPE_TO_TYPE<VECTOR>{typedef float TYPE;};
template<> struct ETYPE_TO_TYPE<FLOAT>{typedef float TYPE;};
template<> struct ETYPE_TO_TYPE<INT>{typedef int TYPE;};
template<> struct ETYPE_TO_TYPE<INDEXEDSTR>{typedef int TYPE;};

template<class T1,class T2> struct
IS_SAME{static const bool value=false;};
template<class T> struct IS_SAME<T,T>{static const bool value=true;};

template<class T> bool
typeCheck(const ParticleAttributeType& type)
{
    // if T is void, don't bother checking what we passed in
    if (IS_SAME<T,void>::value) return true;
    switch(type){
        case VECTOR: return IS_SAME<typename ETYPE_TO_TYPE<VECTOR>::TYPE,T>::value;
        case FLOAT: return IS_SAME<typename ETYPE_TO_TYPE<FLOAT>::TYPE,T>::value;
        case INT: return IS_SAME<typename ETYPE_TO_TYPE<INT>::TYPE,T>::value;
        case INDEXEDSTR: return IS_SAME<typename ETYPE_TO_TYPE<INDEXEDSTR>::TYPE,T>::value;
        default: return false; // unknown type
    }
}

inline 
int TypeSize(ParticleAttributeType attrType)
{
    switch(attrType){
        case NONE: return 0;
        case VECTOR: return sizeof(float);
        case FLOAT: return sizeof(float);
        case INT: return sizeof(int);
        case INDEXEDSTR: return sizeof(int);
        default: return 0;
    }
}

std::string TypeName(ParticleAttributeType attrType);

// Particle Attribute Specifier
//!  Particle Collection Interface
/*!
  This class provides a handle and description of an attribute. This includes
  what type the primitive is, how many instances of the primitive there, name of
  the attribute and an index which speeds lookups of data
*/
class ParticleAttribute
{
public:
    //! Type of attribute
    ParticleAttributeType type;

    //! Number of entries, should be 3 if type is VECTOR
    int count;

    //! Name of attribute
    std::string name;

    //! Internal method of fast access, user should not use or change
    int attributeIndex;

    //! Comment used by various data/readers for extra attribute information
    //! for example for a PTC file to read and write this could be "color" or "point"
    // std::string comment;
};
}
#endif
