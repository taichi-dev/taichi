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

#ifndef _partioendian_h_
#define _partioendian_h_

#include <cassert>
#include <iostream>

namespace Partio{

#ifdef PartioBIG_ENDIAN
static const bool big_endian=true;
#else
static const bool big_endian=false;
#endif

template<class T>
void endianSwap(T& value)
{
     T temp=value;
     char* src=(char*)&temp;
     char* dest=(char*)&value;
     for(unsigned int i=0;i<sizeof(T);i++){
         dest[i]=src[sizeof(T)-i-1];
     }
}

struct BIGEND {
    template<class T> static void swap(T& x){
        if(!big_endian) endianSwap(x);
    }
};

struct LITEND {
    template<class T> static void swap(T& x){
        if(big_endian) endianSwap(x);
    }
};

template<class E,class T> inline void
read(std::istream& input,T& d)
{
    input.read((char*)&d,sizeof(T));
    E::swap(d);
}

template<class E,class T> inline void
write(std::ostream& output,const T& d)
{
    T copy=d;
    E::swap(copy);
    output.write((char*)&copy,sizeof(T));
}

template<class E,class T1,class T2>
void read(std::istream& input,T1& d1,T2& d2)
{read<E>(input,d1);read<E>(input,d2);}

template<class E,class T1,class T2,class T3>
void read(std::istream& input,T1& d1,T2& d2,T3& d3)
{read<E>(input,d1);read<E>(input,d2,d3);}

template<class E,class T1,class T2,class T3,class T4>
void read(std::istream& input,T1& d1,T2& d2,T3& d3,T4& d4)
{read<E>(input,d1);read<E>(input,d2,d3,d4);}

template<class E,class T1,class T2,class T3,class T4,class T5>
void read(std::istream& input,T1& d1,T2& d2,T3& d3,T4& d4,T5& d5)
{read<E>(input,d1);read<E>(input,d2,d3,d4,d5);}

template<class E,class T1,class T2,class T3,class T4,class T5,class T6>
void read(std::istream& input,T1& d1,T2& d2,T3& d3,T4& d4,T5& d5,T6& d6)
{read<E>(input,d1);read<E>(input,d2,d3,d4,d5,d6);}

template<class E,class T1,class T2>
void write(std::ostream& output,const T1& d1,const T2& d2)
{write<E>(output,d1);write<E>(output,d2);}

template<class E,class T1,class T2,class T3>
void write(std::ostream& output,const T1& d1,const T2& d2,const T3& d3)
{write<E>(output,d1);write<E>(output,d2,d3);}

template<class E,class T1,class T2,class T3,class T4>
void write(std::ostream& output,const T1& d1,const T2& d2,const T3& d3,const T4& d4)
{write<E>(output,d1);write<E>(output,d2,d3,d4);}

template<class E,class T1,class T2,class T3,class T4,class T5>
void write(std::ostream& output,const T1& d1,const T2& d2,const T3& d3,const T4& d4,const T5& d5)
{write<E>(output,d1);write<E>(output,d2,d3,d4,d5);}

template<class E,class T1,class T2,class T3,class T4,class T5,class T6>
void write(std::ostream& output,const T1& d1,const T2& d2,const T3& d3,const T4& d4,const T5& d5,const T6& d6)
{write<E>(output,d1);write<E>(output,d2,d3,d4,d5,d6);}

}
#endif
