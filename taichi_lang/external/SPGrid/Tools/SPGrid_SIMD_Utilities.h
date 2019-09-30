//#####################################################################
// Copyright 2018, Haixiang Liu, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Classes SPGrid_Block_Iterator/SPGrid_Reverse_Block_Iterator
//#####################################################################
#ifndef __SPGrid_SIMD_Utilities_h__
#define __SPGrid_SIMD_Utilities_h__

#include <type_traits>
#include <iostream>
#include <cstdint>

namespace SPGrid{

template<typename T,int width> struct SIMD_type;
struct SIMD_type<float,8>  { using type=__m256;  using int_type=__m256i;};
struct SIMD_type<float,16> { using type=__m512;  using int_type=__m512i;};
struct SIMD_type<double,4> { using type=__m256d; using int_type=__m256i;};
struct SIMD_type<double,8> { using type=__m512d; using int_type=__m512i;};

template<typename T,int width> struct SIMD_Operations;
struct SIMD_Operations<float,8>
{
    static constexpr int width = 8;    
    using T = float;
    using type=SIMD_type<T,width>::type;
    using int_type=SIMD_type<T,width>::int_type;
    __forceinline static type zero()                 {return _mm256_setzero_ps();}
    __forceinline static type load (const T* address){return _mm256_load_ps (address);}
    __forceinline static type loadu(const T* address){return _mm256_loadu_ps(address);}
    __forceinline static type set(T A)               {return _mm256_set1_ps(A);}
    __forceinline static type mul(type A,type B)     {return _mm256_mul_ps(A,B);}
    __forceinline static type mul(type A,T B)        {return _mm256_mul_ps(A,_mm256_set1_ps(B));}
    __forceinline static type add(type A,type B)     {return _mm256_add_ps(A,B);}
    __forceinline static type sub(type A,type B)     {return _mm256_sub_ps(A,B);}
    __forceinline static type div(type A,type B)     {return _mm256_div_ps(A,B);}
    __forceinline static type fmadd(type A,type B,type C) {return _mm256_fmadd_ps(A,B,C);}//A*B+C
    __forceinline static type fmadd(type A,T B,type C) {return _mm256_fmadd_ps(A,_mm256_set1_ps(B),C);}//A*B+C
    __forceinline static void store(T* address, type A){_mm256_store_ps(address,A);}
    __forceinline static type blend(int imm8, type A, type B){return _mm256_blend_ps(A,B,imm8);}
    __forceinline static int_type load(const int_type* address){return _mm256_stream_load_si256(address);}
    __forceinline static int_type set(uint64_t A)               {return _mm256_set1_epi64x(A);}
    __forceinline static int_type set(uint32_t A)               {return _mm256_set1_epi32(A);}
    __forceinline static int_type andv(int_type A,int_type B)   {return _mm256_and_si256(A,B);}    
    __forceinline static type blendv(type A, type B,int_type mask){return _mm256_blendv_ps(A,B,_mm256_castsi256_ps(_mm256_cmpgt_epi32(mask,_mm256_setzero_si256())));}
    //__forceinline static type blendv(type A, type B,int_type mask){return _mm256_blendv_ps(A,B,_mm256_castsi256_ps(mask));}
};
   

struct SIMD_Operations<double,4>
{
    static constexpr int width = 4;    
    using T = double;
    using type=SIMD_type<T,width>::type;
    using int_type=SIMD_type<T,width>::int_type;
    __forceinline static type zero()                 {return _mm256_setzero_pd();}
    __forceinline static type load (const T* address){return _mm256_load_pd (address);}
    __forceinline static type loadu(const T* address){return _mm256_loadu_pd(address);}
    __forceinline static type set(T A)               {return _mm256_set1_pd(A);}
    __forceinline static type mul(type A,type B)     {return _mm256_mul_pd(A,B);}
    __forceinline static type mul(type A,T B)        {return _mm256_mul_pd(A,_mm256_set1_pd(B));}
    __forceinline static type add(type A,type B)     {return _mm256_add_pd(A,B);}
    __forceinline static type sub(type A,type B)     {return _mm256_sub_pd(A,B);}
    __forceinline static type div(type A,type B)     {return _mm256_div_pd(A,B);}
    __forceinline static type fmadd(type A,type B,type C) {return _mm256_fmadd_pd(A,B,C);}
    __forceinline static type fmadd(type A,T B,type C) {return _mm256_fmadd_pd(A,_mm256_set1_pd(B),C);}//A*B+C
    __forceinline static void store(T* address, type A){_mm256_store_pd(address,A);}
    __forceinline static type blend(int imm4, type A, type B){return _mm256_blend_pd(A,B,imm4);}
    __forceinline static int_type load(const int_type* address){return _mm256_stream_load_si256(address);}
    __forceinline static int_type set(uint64_t A)               {return _mm256_set1_epi64x(A);}
    __forceinline static int_type set(uint32_t A)               {return _mm256_set1_epi32(A);}
    __forceinline static int_type andv(int_type A,int_type B)   {return _mm256_and_si256(A,B);}    
    //__forceinline static type blendv(type A, type B,int_type mask){return _mm256_blendv_pd(A,B,_mm256_castsi256_pd(mask));}
    __forceinline static type blendv(type A, type B,int_type mask){return _mm256_blendv_pd(A,B,_mm256_castsi256_pd(_mm256_cmpgt_epi64(mask,_mm256_setzero_si256())));}
};
    
struct SIMD_Operations<float,16>
{
    static constexpr int width = 16;    
    using T = float;
    using type=SIMD_type<T,width>::type;
    using int_type=SIMD_type<T,width>::int_type;
    __forceinline static type zero()                 {return _mm512_setzero_ps();}
    __forceinline static type load (const T* address){return _mm512_load_ps (address);}
    __forceinline static type loadu(const T* address){return _mm512_loadu_ps(address);}
    __forceinline static type set(T A)               {return _mm512_set1_ps(A);}
    __forceinline static type mul(type A,type B)     {return _mm512_mul_ps(A,B);}
    __forceinline static type mul(type A,T B)        {return _mm512_mul_ps(A,_mm512_set1_ps(B));}
    __forceinline static type add(type A,type B)     {return _mm512_add_ps(A,B);}
    __forceinline static type sub(type A,type B)     {return _mm512_sub_ps(A,B);}
    __forceinline static type div(type A,type B)     {return _mm512_div_ps(A,B);}
    __forceinline static type fmadd(type A,type B,type C) {return _mm512_fmadd_ps(A,B,C);}
    __forceinline static type fmadd(type A,T B,type C) {return _mm512_fmadd_ps(A,_mm512_set1_ps(B),C);}//A*B+C
    __forceinline static void store(T* address, type A){_mm512_store_ps(address,A);}
    __forceinline static type blend(int imm16, type A, type B){return _mm512_mask_blend_ps(_mm512_int2mask(imm16),A,B);}
    __forceinline static int_type load(const int_type* address){return _mm512_stream_load_si512(address);}
    __forceinline static int_type set(uint64_t A)               {return _mm512_set1_epi64(A);}
    __forceinline static int_type set(uint32_t A)               {return _mm512_set1_epi32(A);}
    __forceinline static int_type andv(int_type A,int_type B)   {return _mm512_and_si512(A,B);}    
    __forceinline static type blendv(type A, type B,int_type mask){return _mm512_mask_blend_ps(_mm512_cmple_epi32_mask(mask,
                                                                                                                       _mm512_setzero_si512()),
                                                                                               B,A);}
};

struct SIMD_Operations<double,8>
{
    static constexpr int width = 8;    
    using T = double;
    using type=SIMD_type<T,width>::type;
    using int_type=SIMD_type<T,width>::int_type;
    __forceinline static type zero()                 {return _mm512_setzero_pd();}
    __forceinline static type load (const T* address){return _mm512_load_pd (address);}
    __forceinline static type loadu(const T* address){return _mm512_loadu_pd(address);}
    __forceinline static type set(T A)               {return _mm512_set1_pd(A);}
    __forceinline static type mul(type A,type B)     {return _mm512_mul_pd(A,B);}
    __forceinline static type mul(type A,T B)        {return _mm512_mul_pd(A,_mm512_set1_pd(B));}
    __forceinline static type add(type A,type B)     {return _mm512_add_pd(A,B);}
    __forceinline static type sub(type A,type B)     {return _mm512_sub_pd(A,B);}
    __forceinline static type div(type A,type B)     {return _mm512_div_pd(A,B);}
    __forceinline static type fmadd(type A,type B,type C) {return _mm512_fmadd_pd(A,B,C);}
    __forceinline static type fmadd(type A,T B,type C) {return _mm512_fmadd_pd(A,_mm512_set1_pd(B),C);}//A*B+C
    __forceinline static void store(T* address, type A){_mm512_store_pd(address,A);}
    __forceinline static type blend(int imm8, type A, type B){return _mm512_mask_blend_pd(_mm512_int2mask(imm8),A,B);}
    __forceinline static int_type load(const int_type* address){return _mm512_stream_load_si512(address);}
    __forceinline static int_type set(uint64_t A)               {return _mm512_set1_epi64(A);}
    __forceinline static int_type set(uint32_t A)               {return _mm512_set1_epi32(A);}
    __forceinline static int_type andv(int_type A,int_type B)   {return _mm512_and_si512(A,B);}
    __forceinline static type blendv(type A, type B,int_type mask){return _mm512_mask_blend_pd(_mm512_cmple_epi64_mask(mask,
                                                                                                                       _mm512_setzero_si512()),
                                                                                               B,A);}
};
        
template<typename T,int width,int zspan> struct SIMD_Blend_Mask;

struct SIMD_Blend_Mask<float,8,4>
{
    static constexpr int z_plus_mask  = 0x77;
    static constexpr int z_minus_mask = 0xee;
    static constexpr int y_plus_mask  = 0x0f;
    static constexpr int y_minus_mask = 0xf0;
};

struct SIMD_Blend_Mask<float,8,8>
{
    static constexpr int z_plus_mask  = 0x7f;
    static constexpr int z_minus_mask = 0xfe;
    static constexpr int y_plus_mask  = 0x00;
    static constexpr int y_minus_mask = 0x00;
};
    
struct SIMD_Blend_Mask<double,4,4>
{
    static constexpr int z_plus_mask  = 0x7;
    static constexpr int z_minus_mask = 0xe;
    static constexpr int y_plus_mask  = 0x0;
    static constexpr int y_minus_mask = 0x0;
};

struct SIMD_Blend_Mask<double,4,8>
{
    static constexpr int z_plus_mask  = 0x7;
    static constexpr int z_minus_mask = 0xe;
    static constexpr int y_plus_mask  = 0x0;
    static constexpr int y_minus_mask = 0x0;
};

struct SIMD_Blend_Mask<float,16,4>
{
    static constexpr int z_plus_mask  = 0x7777;
    static constexpr int z_minus_mask = 0xeeee;
    static constexpr int y_plus_mask  = 0x0fff;
    static constexpr int y_minus_mask = 0xfff0;
};

struct SIMD_Blend_Mask<float,16,8>
{
    static constexpr int z_plus_mask  = 0x7f7f;
    static constexpr int z_minus_mask = 0xfefe;
    static constexpr int y_plus_mask  = 0x00ff;
    static constexpr int y_minus_mask = 0xff00;
};

struct SIMD_Blend_Mask<double,8,4>
{
    static constexpr int z_plus_mask  = 0x77;
    static constexpr int z_minus_mask = 0xee;
    static constexpr int y_plus_mask  = 0x0f;
    static constexpr int y_minus_mask = 0xf0;
};

struct SIMD_Blend_Mask<double,8,8>
{
    static constexpr int z_plus_mask  = 0x7f;
    static constexpr int z_minus_mask = 0xfe;
    static constexpr int y_plus_mask  = 0x00;
    static constexpr int y_minus_mask = 0x00;
};
    
template<typename T,int width>
struct SPGrid_SIMD_Utilities
{
    using Vector_Type       = typename SIMD_type<T,width>::type;
    using Vector_Operations = SIMD_Operations<T,width>;

    template<int di,int dj,int dk,typename T_ARRAY>
    __forceinline static Vector_Type Get_Vector(const uint64_t base_offset,T_ARRAY array)
    {
        static_assert(std::is_same<typename std::remove_cv<typename T_ARRAY::DATA>::type,T>::value,"Type mismatch");
        using T_MASK=typename T_ARRAY::MASK;
        static constexpr int block_ybits=T_MASK::block_ybits;
        static constexpr int block_zbits=T_MASK::block_zbits;
        static constexpr int block_zsize=1<<block_zbits;
        static constexpr int y_span=(width/block_zsize)?(width/block_zsize):1;//number of lines
        static constexpr int z_span=(width>block_zsize)?block_zsize:width;
        
        static_assert((1<<(block_ybits+block_zbits))>=width,"YZ-slice of SPGrid should be multiple of SIMD line");
        Vector_Type result;
        constexpr int zmask=(dk==1)?
            SIMD_Blend_Mask<T,width,z_span>::z_plus_mask:
            SIMD_Blend_Mask<T,width,z_span>::z_minus_mask;
        constexpr int ymask=(dj==1)?
            SIMD_Blend_Mask<T,width,z_span>::y_plus_mask:
            SIMD_Blend_Mask<T,width,z_span>::y_minus_mask;
        const uint64_t x_base_offset=T_ARRAY::MASK::Packed_OffsetXdim<di>(base_offset);
        constexpr uint64_t z_neighbor_padding=dk*(z_span-1)*sizeof(T);
        constexpr uint64_t y_neighbor_padding=dj*(y_span-1)*sizeof(T)*z_span;
        if(dj == 0 && dk == 0)
            result=Vector_Operations::load(&array(x_base_offset));
        else if(dj == 0){
            constexpr uint64_t center_offset=dk*sizeof(T);
            Vector_Type center=Vector_Operations::loadu(&array(x_base_offset+center_offset));
            Vector_Type neighbor=Vector_Operations::loadu(&array(T_MASK::Packed_OffsetZdim<z_span*dk>(x_base_offset)-z_neighbor_padding));
            result=Vector_Operations::blend(zmask,neighbor,center);
        }else if(dk == 0){
            Vector_Type center;
            constexpr uint64_t center_offset=dj*sizeof(T)*z_span;
            if(ymask!=0x00) //Give it a chance to skip the load
                center=Vector_Operations::loadu(&array(x_base_offset+center_offset));
            Vector_Type neighbor=Vector_Operations::loadu(&array(T_MASK::Packed_OffsetYdim<y_span*dj>(x_base_offset)-y_neighbor_padding));
            if(ymask!=0x00){
                result=Vector_Operations::blend(ymask,neighbor,center);
            }else{
                result=neighbor;}
        }else{
            Vector_Type y_center;
            Vector_Type y_neighbor;            
            if(ymask!=0x00){
                constexpr uint64_t center_offset=dk*sizeof(T)+dj*sizeof(T)*z_span;
                constexpr uint64_t neighbor_offset=-z_neighbor_padding+dj*sizeof(T)*z_span;
                Vector_Type center=Vector_Operations::loadu(&array(x_base_offset+center_offset));
                Vector_Type neighbor=Vector_Operations::loadu(&array(T_MASK::Packed_OffsetZdim<z_span*dk>(x_base_offset)
                                                                     +neighbor_offset));
                y_center=Vector_Operations::blend(zmask,neighbor,center);}
            {
                const uint64_t xy_base_offset=T_MASK::Packed_OffsetYdim<y_span*dj>(x_base_offset);
                constexpr uint64_t center_offset=dk*sizeof(T)-y_neighbor_padding;
                constexpr uint64_t neighbor_offset=-z_neighbor_padding-y_neighbor_padding;
                Vector_Type center=Vector_Operations::loadu(&array(xy_base_offset+center_offset));
                Vector_Type neighbor=Vector_Operations::loadu(&array(T_MASK::Packed_OffsetZdim<z_span*dk>(xy_base_offset)
                                                                     +neighbor_offset));
                y_neighbor=Vector_Operations::blend(zmask,neighbor,center);
            }
            if(ymask!=0x00){                
                result=Vector_Operations::blend(ymask,y_neighbor,y_center);
            }else{
                result=y_neighbor;
            }
        }
        return result;
    }
    template<int di,int dj,int dk,typename T_ARRAY>
    __forceinline static void Get_Vector(const uint64_t base_offset,T_ARRAY array_x,T_ARRAY array_y,
                                         Vector_Type& result_x,Vector_Type& result_y)
    {
        static_assert(std::is_same<typename std::remove_cv<typename T_ARRAY::DATA>::type,T>::value,"Type mismatch");
        using T_MASK=typename T_ARRAY::MASK;
        static constexpr int block_ybits=T_MASK::block_ybits;
        static constexpr int block_zbits=T_MASK::block_zbits;
        static constexpr int block_zsize=1<<block_zbits;
        static constexpr int y_span=(width/block_zsize)?(width/block_zsize):1;//number of lines
        static constexpr int z_span=(width>block_zsize)?block_zsize:width;
        
        static_assert((1<<(block_ybits+block_zbits))>=width,"YZ-slice of SPGrid should be multiple of SIMD line");
        constexpr int zmask=(dk==1)?
            SIMD_Blend_Mask<T,width,z_span>::z_plus_mask:
            SIMD_Blend_Mask<T,width,z_span>::z_minus_mask;
        constexpr int ymask=(dj==1)?
            SIMD_Blend_Mask<T,width,z_span>::y_plus_mask:
            SIMD_Blend_Mask<T,width,z_span>::y_minus_mask;
        const uint64_t x_base_offset=T_ARRAY::MASK::Packed_OffsetXdim<di>(base_offset);
        constexpr uint64_t z_neighbor_padding=dk*(z_span-1)*sizeof(T);
        constexpr uint64_t y_neighbor_padding=dj*(y_span-1)*sizeof(T)*z_span;
        if(dj == 0 && dk == 0){
            result_x=Vector_Operations::load(&array_x(x_base_offset));
            result_y=Vector_Operations::load(&array_y(x_base_offset));
        }else if(dj == 0){
            constexpr uint64_t center_offset=dk*sizeof(T);
            const uint64_t center_vector_offset=x_base_offset+center_offset;
            const uint64_t neighbor_vector_offset=T_MASK::Packed_OffsetZdim<z_span*dk>(x_base_offset)-z_neighbor_padding;            
            Vector_Type center_x=Vector_Operations::loadu(&array_x(center_vector_offset));
            Vector_Type center_y=Vector_Operations::loadu(&array_y(center_vector_offset));
            Vector_Type neighbor_x=Vector_Operations::loadu(&array_x(neighbor_vector_offset));
            Vector_Type neighbor_y=Vector_Operations::loadu(&array_y(neighbor_vector_offset));
            result_x=Vector_Operations::blend(zmask,neighbor_x,center_x);
            result_y=Vector_Operations::blend(zmask,neighbor_y,center_y);
        }else if(dk == 0){
            Vector_Type center_x,center_y;
            constexpr uint64_t center_offset=dj*sizeof(T)*z_span;
            const uint64_t center_vector_offset=x_base_offset+center_offset;
            const uint64_t neighbor_vector_offset=T_MASK::Packed_OffsetYdim<y_span*dj>(x_base_offset)-y_neighbor_padding;
            if(ymask!=0x00) {//Give it a chance to skip the load
                center_x=Vector_Operations::loadu(&array_x(x_base_offset+center_offset));
                center_y=Vector_Operations::loadu(&array_y(x_base_offset+center_offset));
            }
            Vector_Type neighbor_x=Vector_Operations::loadu(&array_x(neighbor_vector_offset));
            Vector_Type neighbor_y=Vector_Operations::loadu(&array_y(neighbor_vector_offset));
            if(ymask==0x00) {//Give it a chance to skip the blend                
                result_x=neighbor_x;
                result_y=neighbor_y;
            }else{
                result_x=Vector_Operations::blend(ymask,neighbor_x,center_x);
                result_y=Vector_Operations::blend(ymask,neighbor_y,center_y);
            }
        }else{
            Vector_Type y_center_x,y_center_y;
            Vector_Type y_neighbor_x,y_neighbor_y;            
            if(ymask!=0x00){
                constexpr uint64_t center_offset=dk*sizeof(T)+dj*sizeof(T)*z_span;
                constexpr uint64_t neighbor_offset=-z_neighbor_padding+dj*sizeof(T)*z_span;
                const uint64_t center_vector_offset=x_base_offset+center_offset;
                const uint64_t neighbor_vector_offset=T_MASK::Packed_OffsetZdim<z_span*dk>(x_base_offset)+neighbor_offset;
                Vector_Type center_x=Vector_Operations::loadu(&array_x(center_vector_offset));
                Vector_Type center_y=Vector_Operations::loadu(&array_y(center_vector_offset));
                Vector_Type neighbor_x=Vector_Operations::loadu(&array_x(neighbor_vector_offset));
                Vector_Type neighbor_y=Vector_Operations::loadu(&array_y(neighbor_vector_offset));
                y_center_x=Vector_Operations::blend(zmask,neighbor_x,center_x);
                y_center_y=Vector_Operations::blend(zmask,neighbor_y,center_y);}
            {
                const uint64_t xy_base_offset=T_MASK::Packed_OffsetYdim<y_span*dj>(x_base_offset);
                constexpr uint64_t center_offset=dk*sizeof(T)-y_neighbor_padding;
                constexpr uint64_t neighbor_offset=-z_neighbor_padding-y_neighbor_padding;
                const uint64_t center_vector_offset=xy_base_offset+center_offset;
                const uint64_t neighbor_vector_offset=T_MASK::Packed_OffsetZdim<z_span*dk>(xy_base_offset)+neighbor_offset;
                Vector_Type center_x=Vector_Operations::loadu(&array_x(center_vector_offset));
                Vector_Type center_y=Vector_Operations::loadu(&array_y(center_vector_offset));
                Vector_Type neighbor_x=Vector_Operations::loadu(&array_x(neighbor_vector_offset));
                Vector_Type neighbor_y=Vector_Operations::loadu(&array_y(neighbor_vector_offset));
                y_neighbor_x=Vector_Operations::blend(zmask,neighbor_x,center_x);
                y_neighbor_y=Vector_Operations::blend(zmask,neighbor_y,center_y);
            }
            if(ymask!=0x00){
                result_x=Vector_Operations::blend(ymask,y_neighbor_x,y_center_x); 
                result_y=Vector_Operations::blend(ymask,y_neighbor_y,y_center_y); 
            }else{
                result_x=y_neighbor_x;
                result_y=y_neighbor_y;
            }
        }
    }
    template<int di,int dj,int dk,typename T_ARRAY>
    __forceinline static void Get_Vector(const uint64_t base_offset,T_ARRAY array_x,T_ARRAY array_y,T_ARRAY array_z,
                                         Vector_Type& result_x,Vector_Type& result_y,Vector_Type& result_z)
    {
        static_assert(std::is_same<typename std::remove_cv<typename T_ARRAY::DATA>::type,T>::value,"Type mismatch");
        using T_MASK=typename T_ARRAY::MASK;
        static constexpr int block_ybits=T_MASK::block_ybits;
        static constexpr int block_zbits=T_MASK::block_zbits;
        static constexpr int block_zsize=1<<block_zbits;
        static constexpr int y_span=(width/block_zsize)?(width/block_zsize):1;//number of lines
        static constexpr int z_span=(width>block_zsize)?block_zsize:width;
        
        static_assert((1<<(block_ybits+block_zbits))>=width,"YZ-slice of SPGrid should be multiple of SIMD line");
        constexpr int zmask=(dk==1)?
            SIMD_Blend_Mask<T,width,z_span>::z_plus_mask:
            SIMD_Blend_Mask<T,width,z_span>::z_minus_mask;
        constexpr int ymask=(dj==1)?
            SIMD_Blend_Mask<T,width,z_span>::y_plus_mask:
            SIMD_Blend_Mask<T,width,z_span>::y_minus_mask;
        const uint64_t x_base_offset=T_ARRAY::MASK::Packed_OffsetXdim<di>(base_offset);
        constexpr uint64_t z_neighbor_padding=dk*(z_span-1)*sizeof(T);
        constexpr uint64_t y_neighbor_padding=dj*(y_span-1)*sizeof(T)*z_span;
        if(dj == 0 && dk == 0){
            result_x=Vector_Operations::load(&array_x(x_base_offset));
            result_y=Vector_Operations::load(&array_y(x_base_offset));
            result_z=Vector_Operations::load(&array_z(x_base_offset));
        }else if(dj == 0){
            constexpr uint64_t center_offset=dk*sizeof(T);
            const uint64_t center_vector_offset=x_base_offset+center_offset;
            const uint64_t neighbor_vector_offset=T_MASK::Packed_OffsetZdim<z_span*dk>(x_base_offset)-z_neighbor_padding;            
            Vector_Type center_x=Vector_Operations::loadu(&array_x(center_vector_offset));
            Vector_Type center_y=Vector_Operations::loadu(&array_y(center_vector_offset));
            Vector_Type center_z=Vector_Operations::loadu(&array_z(center_vector_offset));
            Vector_Type neighbor_x=Vector_Operations::loadu(&array_x(neighbor_vector_offset));
            Vector_Type neighbor_y=Vector_Operations::loadu(&array_y(neighbor_vector_offset));
            Vector_Type neighbor_z=Vector_Operations::loadu(&array_z(neighbor_vector_offset));
            result_x=Vector_Operations::blend(zmask,neighbor_x,center_x);
            result_y=Vector_Operations::blend(zmask,neighbor_y,center_y);
            result_z=Vector_Operations::blend(zmask,neighbor_z,center_z);
        }else if(dk == 0){
            Vector_Type center_x,center_y,center_z;
            constexpr uint64_t center_offset=dj*sizeof(T)*z_span;
            const uint64_t center_vector_offset=x_base_offset+center_offset;
            const uint64_t neighbor_vector_offset=T_MASK::Packed_OffsetYdim<y_span*dj>(x_base_offset)-y_neighbor_padding;
            if(ymask!=0x00) {//Give it a chance to skip the load
                center_x=Vector_Operations::loadu(&array_x(x_base_offset+center_offset));
                center_y=Vector_Operations::loadu(&array_y(x_base_offset+center_offset));
                center_z=Vector_Operations::loadu(&array_z(x_base_offset+center_offset));
            }
            Vector_Type neighbor_x=Vector_Operations::loadu(&array_x(neighbor_vector_offset));
            Vector_Type neighbor_y=Vector_Operations::loadu(&array_y(neighbor_vector_offset));
            Vector_Type neighbor_z=Vector_Operations::loadu(&array_z(neighbor_vector_offset));
            if(ymask==0x00) {//Give it a chance to skip the blend                
                result_x=neighbor_x;
                result_y=neighbor_y;
                result_z=neighbor_z;
            }else{
                result_x=Vector_Operations::blend(ymask,neighbor_x,center_x);
                result_y=Vector_Operations::blend(ymask,neighbor_y,center_y);
                result_z=Vector_Operations::blend(ymask,neighbor_z,center_z);
            }
        }else{
            Vector_Type y_center_x,y_center_y,y_center_z;
            Vector_Type y_neighbor_x,y_neighbor_y,y_neighbor_z;            
            if(ymask!=0x00){
                constexpr uint64_t center_offset=dk*sizeof(T)+dj*sizeof(T)*z_span;
                constexpr uint64_t neighbor_offset=-z_neighbor_padding+dj*sizeof(T)*z_span;
                const uint64_t center_vector_offset=x_base_offset+center_offset;
                const uint64_t neighbor_vector_offset=T_MASK::Packed_OffsetZdim<z_span*dk>(x_base_offset)+neighbor_offset;
                Vector_Type center_x=Vector_Operations::loadu(&array_x(center_vector_offset));
                Vector_Type center_y=Vector_Operations::loadu(&array_y(center_vector_offset));
                Vector_Type center_z=Vector_Operations::loadu(&array_z(center_vector_offset));
                Vector_Type neighbor_x=Vector_Operations::loadu(&array_x(neighbor_vector_offset));
                Vector_Type neighbor_y=Vector_Operations::loadu(&array_y(neighbor_vector_offset));
                Vector_Type neighbor_z=Vector_Operations::loadu(&array_z(neighbor_vector_offset));
                y_center_x=Vector_Operations::blend(zmask,neighbor_x,center_x);
                y_center_y=Vector_Operations::blend(zmask,neighbor_y,center_y);
                y_center_z=Vector_Operations::blend(zmask,neighbor_z,center_z);}
            {
                const uint64_t xy_base_offset=T_MASK::Packed_OffsetYdim<y_span*dj>(x_base_offset);
                constexpr uint64_t center_offset=dk*sizeof(T)-y_neighbor_padding;
                constexpr uint64_t neighbor_offset=-z_neighbor_padding-y_neighbor_padding;
                const uint64_t center_vector_offset=xy_base_offset+center_offset;
                const uint64_t neighbor_vector_offset=T_MASK::Packed_OffsetZdim<z_span*dk>(xy_base_offset)+neighbor_offset;
                Vector_Type center_x=Vector_Operations::loadu(&array_x(center_vector_offset));
                Vector_Type center_y=Vector_Operations::loadu(&array_y(center_vector_offset));
                Vector_Type center_z=Vector_Operations::loadu(&array_z(center_vector_offset));
                Vector_Type neighbor_x=Vector_Operations::loadu(&array_x(neighbor_vector_offset));
                Vector_Type neighbor_y=Vector_Operations::loadu(&array_y(neighbor_vector_offset));
                Vector_Type neighbor_z=Vector_Operations::loadu(&array_z(neighbor_vector_offset));
                y_neighbor_x=Vector_Operations::blend(zmask,neighbor_x,center_x);
                y_neighbor_y=Vector_Operations::blend(zmask,neighbor_y,center_y);
                y_neighbor_z=Vector_Operations::blend(zmask,neighbor_z,center_z);
            }
            if(ymask!=0x00){
                result_x=Vector_Operations::blend(ymask,y_neighbor_x,y_center_x); 
                result_y=Vector_Operations::blend(ymask,y_neighbor_y,y_center_y); 
                result_z=Vector_Operations::blend(ymask,y_neighbor_z,y_center_z); 
            }else{
                result_x=y_neighbor_x;
                result_y=y_neighbor_y;
                result_z=y_neighbor_z;
            }
        }
    }
};
}
#endif
