//#####################################################################
// Copyright (c) 2012, Eftychios Sifakis, Sean Bauer.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
// Class SPGrid_Geometry
//#####################################################################
#ifndef __SPGrid_Geometry_h__
#define __SPGrid_Geometry_h__

#include <SPGrid/Core/SPGrid_Utilities.h>
#include <array>
#include <algorithm>

namespace SPGrid{

template<int dim> struct SPGrid_Geometry;

template<>
struct SPGrid_Geometry<3>
{
    const ucoord_t xsize,ysize,zsize;                      // Dimensions requested
    const ucoord_t block_xsize,block_ysize,block_zsize;    // Dimensions of data layout within block
    const ucoord_t zsize_padded,ysize_padded,xsize_padded; // Dimensions allocated; adjusted for alignment and uniformity
    
    SPGrid_Geometry(const ucoord_t xsize_input,const ucoord_t ysize_input,const ucoord_t zsize_input,
        const int block_xbits,const int block_ybits,const int block_zbits)
        :xsize(xsize_input),ysize(ysize_input),zsize(zsize_input),
        block_xsize(1u<<block_xbits),block_ysize(1u<<block_ybits),block_zsize(1u<<block_zbits),
        zsize_padded(Next_Power_Of_Two(std::max({xsize,ysize,zsize,block_zsize}))),
        ysize_padded(std::max({Next_Power_Of_Two(std::max(ysize,xsize)),zsize_padded>>1,block_ysize})),
        xsize_padded(std::max({Next_Power_Of_Two(xsize),zsize_padded>>1,block_xsize}))
    {}

    std::array<ucoord_t,3> Padded_Size() const
    {return std::array<ucoord_t,3>{{xsize_padded,ysize_padded,zsize_padded}};}
    
    uint64_t Padded_Volume() const
    {return (uint64_t)xsize_padded*(uint64_t)ysize_padded*(uint64_t)zsize_padded;}

    uint32_t Elements_Per_Block() const
    {return block_xsize*block_ysize*block_zsize;}

    std::array<ucoord_t,3> Block_Size() const
    {return std::array<ucoord_t,3>{{block_xsize,block_ysize,block_zsize}};}

//#####################################################################
    void Check_Bounds(const ucoord_t i,const ucoord_t j,const ucoord_t k) const;
//#####################################################################
};

template<>
struct SPGrid_Geometry<2>
{
    const ucoord_t xsize,ysize;               // Dimensions requested
    const ucoord_t block_xsize,block_ysize;   // Dimensions of data layout within block
    const ucoord_t ysize_padded,xsize_padded; // Dimensions allocated; adjusted for alignment and uniformity

    SPGrid_Geometry(const ucoord_t xsize_input,const ucoord_t ysize_input,const int block_xbits,const int block_ybits)
        :xsize(xsize_input),ysize(ysize_input),
        block_xsize(1u<<block_xbits),block_ysize(1u<<block_ybits),
        ysize_padded(Next_Power_Of_Two(std::max({xsize,ysize,block_ysize}))),
        xsize_padded(std::max({Next_Power_Of_Two(xsize),ysize_padded>>1,block_xsize}))
    {}

    std::array<ucoord_t,2> Padded_Size() const
    {return std::array<ucoord_t,2>{{xsize_padded,ysize_padded}};}

    uint64_t Padded_Volume() const
    {return (uint64_t)xsize_padded*(uint64_t)ysize_padded;}
    
    uint32_t Elements_Per_Block() const
    {return block_xsize*block_ysize;}

    std::array<ucoord_t,2> Block_Size() const
    {return std::array<ucoord_t,2>{{block_xsize,block_ysize}};}

//#####################################################################
    void Check_Bounds(const ucoord_t i,const ucoord_t j) const;
//#####################################################################
};
}

#endif
