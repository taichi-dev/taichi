//#####################################################################
// Copyright (c) 2012-2013, Sean Bauer, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
// Utility classes/functions
//! @file
//#####################################################################
#ifndef __SPGrid_Utilities_h__
#define __SPGrid_Utilities_h__

#include <sstream>
#include <immintrin.h>

#define HASWELL

namespace SPGrid{

typedef uint32_t ucoord_t;
typedef int32_t scoord_t;

//! @struct BitLength
//! Class BitLength computes the bit-length of its template parameter at compile time. It is used primarily to quantize the size of struct's passed to SPGrid_Allocator to the next
//! power of two.
//! @brief Bit-length of integer template argument
//! @tparam d Integer input whose bit length is to be computed

template<int d> struct BitLength;
template<> struct BitLength<0> {enum {value=0};};
template<int d> struct BitLength {enum {value/*!< Bit length of template parameter */ =1+BitLength<(d>>1)>::value};};
template<int d> struct NextLogTwo {enum {value=BitLength<d-1>::value};};

//! Computes the byte offset of a member within a class. The member is specified by means of a pointer-to-member. For example, if seeking the byte offset of float-valued member
//! *my_float* within the struct *my_struct*, the argument passed to this function should be \& *my_struct::my_float*.
//! @tparam T_FIELD Type of the member being localized
//! @tparam T A class type containing a member field of type T
//! @param field Pointer to a member (of type T_FIELD) of class T
//! @return Offset, in bytes, of specified struct member
template<class T,class T_FIELD> size_t
OffsetOfMember(T_FIELD T::*field)
{return (size_t)((char*)&(((T*)0)->*field)-(char*)0);}

//! Rounds-up (at run-time) the input parameter to the closest power of two.
//! @param i Integer argument to be rounded up to a power of two.
//! @return Lowest power of two that is greater or equal than the input parameter.
inline uint32_t Next_Power_Of_Two(uint32_t i)
{i--;i|=i>>1;i|=i>>2;i|=i>>4;i|=i>>8;i|=i>>16;return i+1;}

template<class T> std::string Value_To_String(const T& value)
{std::ostringstream output;output<<value;return output.str();}

#ifdef HASWELL
uint64_t Bit_Spread(const ucoord_t data,const uint64_t mask);
uint64_t Bit_Spread(const scoord_t data,const uint64_t mask);
#else
template<uint64_t mask>
inline uint64_t Bit_Spread_u64(uint64_t uldata)
{
    uint64_t result=0;

    if(0x0000000000000001UL & mask) result |= uldata & 0x0000000000000001UL; else uldata <<= 1;
    if(0x0000000000000002UL & mask) result |= uldata & 0x0000000000000002UL; else uldata <<= 1;
    if(0x0000000000000004UL & mask) result |= uldata & 0x0000000000000004UL; else uldata <<= 1;
    if(0x0000000000000008UL & mask) result |= uldata & 0x0000000000000008UL; else uldata <<= 1;
    if(0x0000000000000010UL & mask) result |= uldata & 0x0000000000000010UL; else uldata <<= 1;
    if(0x0000000000000020UL & mask) result |= uldata & 0x0000000000000020UL; else uldata <<= 1;
    if(0x0000000000000040UL & mask) result |= uldata & 0x0000000000000040UL; else uldata <<= 1;
    if(0x0000000000000080UL & mask) result |= uldata & 0x0000000000000080UL; else uldata <<= 1;
    if(0x0000000000000100UL & mask) result |= uldata & 0x0000000000000100UL; else uldata <<= 1;
    if(0x0000000000000200UL & mask) result |= uldata & 0x0000000000000200UL; else uldata <<= 1;
    if(0x0000000000000400UL & mask) result |= uldata & 0x0000000000000400UL; else uldata <<= 1;
    if(0x0000000000000800UL & mask) result |= uldata & 0x0000000000000800UL; else uldata <<= 1;
    if(0x0000000000001000UL & mask) result |= uldata & 0x0000000000001000UL; else uldata <<= 1;
    if(0x0000000000002000UL & mask) result |= uldata & 0x0000000000002000UL; else uldata <<= 1;
    if(0x0000000000004000UL & mask) result |= uldata & 0x0000000000004000UL; else uldata <<= 1;
    if(0x0000000000008000UL & mask) result |= uldata & 0x0000000000008000UL; else uldata <<= 1;
    if(0x0000000000010000UL & mask) result |= uldata & 0x0000000000010000UL; else uldata <<= 1;
    if(0x0000000000020000UL & mask) result |= uldata & 0x0000000000020000UL; else uldata <<= 1;
    if(0x0000000000040000UL & mask) result |= uldata & 0x0000000000040000UL; else uldata <<= 1;
    if(0x0000000000080000UL & mask) result |= uldata & 0x0000000000080000UL; else uldata <<= 1;
    if(0x0000000000100000UL & mask) result |= uldata & 0x0000000000100000UL; else uldata <<= 1;
    if(0x0000000000200000UL & mask) result |= uldata & 0x0000000000200000UL; else uldata <<= 1;
    if(0x0000000000400000UL & mask) result |= uldata & 0x0000000000400000UL; else uldata <<= 1;
    if(0x0000000000800000UL & mask) result |= uldata & 0x0000000000800000UL; else uldata <<= 1;
    if(0x0000000001000000UL & mask) result |= uldata & 0x0000000001000000UL; else uldata <<= 1;
    if(0x0000000002000000UL & mask) result |= uldata & 0x0000000002000000UL; else uldata <<= 1;
    if(0x0000000004000000UL & mask) result |= uldata & 0x0000000004000000UL; else uldata <<= 1;
    if(0x0000000008000000UL & mask) result |= uldata & 0x0000000008000000UL; else uldata <<= 1;
    if(0x0000000010000000UL & mask) result |= uldata & 0x0000000010000000UL; else uldata <<= 1;
    if(0x0000000020000000UL & mask) result |= uldata & 0x0000000020000000UL; else uldata <<= 1;
    if(0x0000000040000000UL & mask) result |= uldata & 0x0000000040000000UL; else uldata <<= 1;
    if(0x0000000080000000UL & mask) result |= uldata & 0x0000000080000000UL; else uldata <<= 1;
    if(0x0000000100000000UL & mask) result |= uldata & 0x0000000100000000UL; else uldata <<= 1;
    if(0x0000000200000000UL & mask) result |= uldata & 0x0000000200000000UL; else uldata <<= 1;
    if(0x0000000400000000UL & mask) result |= uldata & 0x0000000400000000UL; else uldata <<= 1;
    if(0x0000000800000000UL & mask) result |= uldata & 0x0000000800000000UL; else uldata <<= 1;
    if(0x0000001000000000UL & mask) result |= uldata & 0x0000001000000000UL; else uldata <<= 1;
    if(0x0000002000000000UL & mask) result |= uldata & 0x0000002000000000UL; else uldata <<= 1;
    if(0x0000004000000000UL & mask) result |= uldata & 0x0000004000000000UL; else uldata <<= 1;
    if(0x0000008000000000UL & mask) result |= uldata & 0x0000008000000000UL; else uldata <<= 1;
    if(0x0000010000000000UL & mask) result |= uldata & 0x0000010000000000UL; else uldata <<= 1;
    if(0x0000020000000000UL & mask) result |= uldata & 0x0000020000000000UL; else uldata <<= 1;
    if(0x0000040000000000UL & mask) result |= uldata & 0x0000040000000000UL; else uldata <<= 1;
    if(0x0000080000000000UL & mask) result |= uldata & 0x0000080000000000UL; else uldata <<= 1;
    if(0x0000100000000000UL & mask) result |= uldata & 0x0000100000000000UL; else uldata <<= 1;
    if(0x0000200000000000UL & mask) result |= uldata & 0x0000200000000000UL; else uldata <<= 1;
    if(0x0000400000000000UL & mask) result |= uldata & 0x0000400000000000UL; else uldata <<= 1;
    if(0x0000800000000000UL & mask) result |= uldata & 0x0000800000000000UL; else uldata <<= 1;
    if(0x0001000000000000UL & mask) result |= uldata & 0x0001000000000000UL; else uldata <<= 1;
    if(0x0002000000000000UL & mask) result |= uldata & 0x0002000000000000UL; else uldata <<= 1;
    if(0x0004000000000000UL & mask) result |= uldata & 0x0004000000000000UL; else uldata <<= 1;
    if(0x0008000000000000UL & mask) result |= uldata & 0x0008000000000000UL; else uldata <<= 1;
    if(0x0010000000000000UL & mask) result |= uldata & 0x0010000000000000UL; else uldata <<= 1;
    if(0x0020000000000000UL & mask) result |= uldata & 0x0020000000000000UL; else uldata <<= 1;
    if(0x0040000000000000UL & mask) result |= uldata & 0x0040000000000000UL; else uldata <<= 1;
    if(0x0080000000000000UL & mask) result |= uldata & 0x0080000000000000UL; else uldata <<= 1;
    if(0x0100000000000000UL & mask) result |= uldata & 0x0100000000000000UL; else uldata <<= 1;
    if(0x0200000000000000UL & mask) result |= uldata & 0x0200000000000000UL; else uldata <<= 1;
    if(0x0400000000000000UL & mask) result |= uldata & 0x0400000000000000UL; else uldata <<= 1;
    if(0x0800000000000000UL & mask) result |= uldata & 0x0800000000000000UL; else uldata <<= 1;
    if(0x1000000000000000UL & mask) result |= uldata & 0x1000000000000000UL; else uldata <<= 1;
    if(0x2000000000000000UL & mask) result |= uldata & 0x2000000000000000UL; else uldata <<= 1;
    if(0x4000000000000000UL & mask) result |= uldata & 0x4000000000000000UL; else uldata <<= 1;
    if(0x8000000000000000UL & mask) result |= uldata & 0x8000000000000000UL; else uldata <<= 1;

    return result;
}
template<uint64_t mask> uint64_t Bit_Spread(const ucoord_t data)
{uint64_t uldata=data;return Bit_Spread_u64<mask>(uldata);}
template<uint64_t mask> uint64_t Bit_Spread(const scoord_t data)
{union{ int64_t sldata; uint64_t uldata; };sldata=data;return Bit_Spread_u64<mask>(uldata);}
#endif

scoord_t Bit_Pack(const uint64_t data, const uint64_t mask);

template<uint64_t data,uint64_t mask,uint64_t bit=1UL> struct BitSpread;
template<uint64_t data,uint64_t mask> struct BitSpread<data,mask,0> { static const uint64_t value=0; };
template<uint64_t data,uint64_t mask,uint64_t bit> struct BitSpread
{ static const uint64_t value = (bit & mask ) ? BitSpread<data,mask,bit<<1>::value | (data & bit) : BitSpread<data<<1,mask,bit<<1>::value; };

#define FATAL_ERROR(...) \
    Fatal_Error((const char*)__FUNCTION__,__FILE__,__LINE__,##__VA_ARGS__)

void Fatal_Error(const char* function,const char* file,int line) __attribute__ ((noreturn)) ;
void Fatal_Error(const char* function,const char* file,int line,const char* message) __attribute__ ((noreturn)) ;
void Fatal_Error(const char* function,const char* file,int line,const std::string& message) __attribute__ ((noreturn)) ;

// Used during allocation
void Check_Compliance();
void* Raw_Allocate(const size_t size);
void Raw_Deallocate(void* data, const size_t size);
void Deactivate_Page(void* data, const size_t size);

// Used to check address resident
void Check_Address_Resident(const void* addr);

// Check whether physically-mapped pages are correctly registered
void Validate_Memory_Use(uint64_t number_of_pages,void *data_ptr,uint64_t *page_mask_array);

}
#endif
