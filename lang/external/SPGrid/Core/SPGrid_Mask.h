//#####################################################################
// Copyright (c) 2012-2013, Sean Bauer, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Mask_h__
#define __SPGrid_Mask_h__

#include <SPGrid/Core/SPGrid_Utilities.h>
#include <array>

namespace SPGrid{

template<int log2_struct,int D,int log2_page=12> class SPGrid_Mask_base;
template<int log2_struct,int log2_field,int D,int log2_page=12> class SPGrid_Mask;

//#####################################################################
// Class SPGrid_Mask_base (3D)
//#####################################################################

template<int log2_struct,int log2_page>
class SPGrid_Mask_base<log2_struct,3,log2_page>
{
    static_assert(log2_page>=12,"Please make sure that the page size is larger than a physical page(4096)");
public:
    enum {
        page_bits=log2_page                        // Bits needed for indexing individual bytes within a page (not necessarily a 4KB page)
    };
protected:

    enum {
        data_bits=log2_struct,                     // Bits needed for indexing individual bytes within type T
        block_bits=page_bits-data_bits,            // Bits needed for indexing data elements within a block
        block_zbits=block_bits/3+(block_bits%3>0), // Bits needed for the z-coordinate of a data elements within a block
        block_ybits=block_bits/3+(block_bits%3>1), // Bits needed for the y-coordinate of a data elements within a block
        block_xbits=block_bits/3                   // Bits needed for the x-coordinate of a data elements within a block
    };

    enum : uint64_t { // Bit masks for the upper 52 bits of memory addresses (page indices)
        // New mask structure, after allowing non-4KB pages
        page_zmask=(0x9249249249249249UL<<(3-block_bits%3+(page_bits-12)))&(0xffffffffffffffffUL<<page_bits),
        page_ymask=(0x2492492492492492UL<<(3-block_bits%3+(page_bits-12)))&(0xffffffffffffffffUL<<page_bits),
        page_xmask=(0x4924924924924924UL<<(3-block_bits%3+(page_bits-12)))&(0xffffffffffffffffUL<<page_bits)
        // Old mask structure, specifically for 4KB pages
        // page_zmask=(0x9249249249249249UL<<(3-block_bits%3))&0xfffffffffffff000UL,
        // page_ymask=(0x2492492492492492UL<<(3-block_bits%3))&0xfffffffffffff000UL,
        // page_xmask=(0x4924924924924924UL<<(3-block_bits%3))&0xfffffffffffff000UL
    };
public:
    enum : uint32_t {elements_per_block=1u<<block_bits};
};

//#####################################################################
// Class SPGrid_Mask (3D)
//#####################################################################

template<int log2_struct,int log2_field,int log2_page>
class SPGrid_Mask<log2_struct,log2_field,3,log2_page>: public SPGrid_Mask_base<log2_struct,3,log2_page>
{
public:
    enum {dim=3};
    enum {field_size=1<<log2_field};
    typedef SPGrid_Mask_base<log2_struct,dim,log2_page> T_Mask_base;
    using T_Mask_base::data_bits;using T_Mask_base::block_bits;using T_Mask_base::page_bits;
    using T_Mask_base::block_xbits;using T_Mask_base::block_ybits;using T_Mask_base::block_zbits;
    using T_Mask_base::elements_per_block;

private:
    using T_Mask_base::page_xmask;using T_Mask_base::page_ymask;using T_Mask_base::page_zmask;    
    enum { // Bit masks for the lower 12 bits of memory addresses (element indices within a page)
        element_zmask=((1<<block_zbits)-1)<<log2_field,                              // Likely ok! (after !=4KB generalization)
        element_ymask=((1<<block_ybits)-1)<<(log2_field+block_zbits),                // Likely ok! (after !=4KB generalization)
        element_xmask=((1<<block_xbits)-1)<<(log2_field+block_zbits+block_ybits)     // Likely ok! (after !=4KB generalization)
    };
    
public:
    
    enum { // This needs to be carefully re-checked
        // Same as the corresponding element bit masks, but with the most significant bit the respective coordinate zeroed out
        element_z_lsbits=(element_zmask>>1)&element_zmask,
        element_y_lsbits=(element_ymask>>1)&element_ymask,
        element_x_lsbits=(element_xmask>>1)&element_xmask,
        
        // Same as the corresponding element bit masks, but with the least significant bit the respective coordinate zeroed out
        element_z_msbits=(element_zmask<<1)&element_zmask,
        element_y_msbits=(element_ymask<<1)&element_ymask,
        element_x_msbits=(element_xmask<<1)&element_xmask,

        // Just the most significant bit of the element bit mask for the respective coordinate
        element_z_msbit=element_zmask^element_z_lsbits,
        element_y_msbit=element_ymask^element_y_lsbits,
        element_x_msbit=element_xmask^element_x_lsbits,
        
        downsample_lower_mask = element_z_lsbits | element_y_lsbits | element_x_lsbits,
        upsample_lower_mask   = element_z_msbits | element_y_msbits | element_x_msbits,
        
        // "Left over bits" - lob=0 means page address starts with z-bit, lob=1 is x-bit, lob=2 is y-bit
        lob = (3 - block_bits%3)%3,

        xloc = lob==0 ? (log2_page+3) : ( lob==1 ? (log2_page+1) : (log2_page+2)),
        yloc = lob==0 ? (log2_page+2) : ( lob==1 ? (log2_page+3) : (log2_page+1)),
        zloc = lob==0 ? (log2_page+1) : ( lob==1 ? (log2_page+2) : (log2_page+3)),

        u_zbit_shift = zloc - (log2_field+block_zbits),
        u_ybit_shift = yloc - (log2_field+block_zbits+block_ybits),
        u_xbit_shift = xloc - (log2_field+block_bits),

        bit_log2_page_mask        = lob==0 ? element_z_msbit : ( lob==1 ? element_x_msbit : element_y_msbit ),
        bit_log2_page_plus_1_mask = lob==0 ? element_y_msbit : ( lob==1 ? element_z_msbit : element_x_msbit ),
        bit_log2_page_plus_2_mask = lob==0 ? element_x_msbit : ( lob==1 ? element_y_msbit : element_z_msbit )

    };
    
    enum { // Bit masks for aggregate addresses
        zmask=page_zmask|(uint64_t)element_zmask,
        ymask=page_ymask|(uint64_t)element_ymask,
        xmask=page_xmask|(uint64_t)element_xmask
    };
    enum { 
        MXADD_Zmask=~zmask, 
        MXADD_Ymask=~ymask, 
        MXADD_Xmask=~xmask, 
        MXADD_Wmask=xmask|ymask|zmask
    };
    enum {
        ODD_BITS=BitSpread<1,xmask>::value | BitSpread<1,ymask>::value | BitSpread<1,zmask>::value
    };


public:

    static unsigned int Bytes_Per_Element()
    {return 1u<<data_bits;}

    static unsigned int Elements_Per_Block()
    {return elements_per_block;}

    template<int i, int j, int k> struct LinearOffset
    {
      static const uint64_t value = BitSpread<(uint64_t)i,xmask>::value | BitSpread<(uint64_t)j,ymask>::value | BitSpread<(uint64_t)k,zmask>::value;
    };

    inline static uint64_t Linear_Offset(const int i, const int j, const int k)
    {
#ifdef HASWELL
        return Bit_Spread(i,xmask)|Bit_Spread(j,ymask)|Bit_Spread(k,zmask);
#else
        return Bit_Spread<xmask>(i)|Bit_Spread<ymask>(j)|Bit_Spread<zmask>(k);
#endif
    }

    inline static uint64_t Linear_Offset(const std::array<int,3>& coord)
    {
#ifdef HASWELL
        return Bit_Spread(coord[0],xmask)|Bit_Spread(coord[1],ymask)|Bit_Spread(coord[2],zmask);
#else
        return Bit_Spread<xmask>(coord[0])|Bit_Spread<ymask>(coord[1])|Bit_Spread<zmask>(coord[2]);
#endif
    }
    inline static uint64_t Linear_Offset(const std::array<unsigned int,3>& coord)
    {
#ifdef HASWELL
        return Bit_Spread(coord[0],xmask)|Bit_Spread(coord[1],ymask)|Bit_Spread(coord[2],zmask);
#else
        return Bit_Spread<xmask>(coord[0])|Bit_Spread<ymask>(coord[1])|Bit_Spread<zmask>(coord[2]);
#endif
    }

    inline static void LinearToCoord(uint64_t linear_offset, int* i, int* j, int* k)
    {
        *i = Bit_Pack(linear_offset,xmask);
        *j = Bit_Pack(linear_offset,ymask);
        *k = Bit_Pack(linear_offset,zmask);
    }
    inline static std::array<int,3> LinearToCoord(uint64_t linear_offset)
    {
        std::array<int,3> coord;
        coord[0] = Bit_Pack(linear_offset,xmask);
        coord[1] = Bit_Pack(linear_offset,ymask);
        coord[2] = Bit_Pack(linear_offset,zmask);
        return coord;
    }

    // 1-based offset calculation
    template <int originX=1,int originY=1,int originZ=1>
    inline static uint64_t DownsampleOffset(uint64_t linear_offset)
    {
        return Packed_RightShift(Packed_Offset<originX,originY,originZ>(linear_offset));
    }


    inline static uint64_t Packed_RightShift(uint64_t linear_offset)
    {
        static uint64_t my_array[8] = {
            0,
            bit_log2_page_mask,
            bit_log2_page_plus_1_mask,
            bit_log2_page_mask|bit_log2_page_plus_1_mask,
            bit_log2_page_plus_2_mask,
            bit_log2_page_mask|bit_log2_page_plus_2_mask,
            bit_log2_page_plus_1_mask|bit_log2_page_plus_2_mask,
            bit_log2_page_mask|bit_log2_page_plus_1_mask|bit_log2_page_plus_2_mask};

        uint64_t upper = (linear_offset >> 3) & (0xffffffffffffffffUL << log2_page);
        uint64_t lower = (linear_offset >> 1) & downsample_lower_mask;
        uint64_t result = upper | lower | my_array[(linear_offset>>log2_page) & 0x7UL];
        return result;
    }
    
    // 1-based offset calculation
    template<int originX=1,int originY=1,int originZ=1>
    inline static uint64_t UpsampleOffset(uint64_t linear_offset)
    {
        return Packed_Offset<-originX,-originY,-originZ>(Packed_LeftShift(linear_offset));
    }

    inline static uint64_t Packed_LeftShift(uint64_t linear_offset)
    {
        uint64_t upper = (linear_offset << 3) & (0xffffffffffffffffUL << log2_page);
        uint64_t lower = (linear_offset << 1) & upsample_lower_mask;

        uint64_t x_bit_place = (linear_offset & element_x_msbit)<<u_xbit_shift;
        uint64_t y_bit_place = (linear_offset & element_y_msbit)<<u_ybit_shift;
        uint64_t z_bit_place = (linear_offset & element_z_msbit)<<u_zbit_shift;

        uint64_t result = upper | lower | x_bit_place | y_bit_place | z_bit_place;
        return result;
    }

    inline static uint64_t Packed_Add(const uint64_t i,const uint64_t j)
    {
        uint64_t x_result=( (i | MXADD_Xmask) + (j & ~MXADD_Xmask) ) & ~MXADD_Xmask;
        uint64_t y_result=( (i | MXADD_Ymask) + (j & ~MXADD_Ymask) ) & ~MXADD_Ymask;
        uint64_t z_result=( (i | MXADD_Zmask) + (j & ~MXADD_Zmask) ) & ~MXADD_Zmask;
        uint64_t w_result=( (i | MXADD_Wmask) + (j & ~MXADD_Wmask) ) & ~MXADD_Wmask;
        uint64_t result=x_result | y_result | z_result | w_result;
        return result;
    }

    template<uint64_t j>
    inline static uint64_t Packed_Add_Imm(const uint64_t i)
    {
        uint64_t x_result=( (i | MXADD_Xmask) + (j & ~MXADD_Xmask) ) & ~MXADD_Xmask;
        uint64_t y_result=( (i | MXADD_Ymask) + (j & ~MXADD_Ymask) ) & ~MXADD_Ymask;
        uint64_t z_result=( (i | MXADD_Zmask) + (j & ~MXADD_Zmask) ) & ~MXADD_Zmask;
        uint64_t w_result=( (i | MXADD_Wmask) + (j & ~MXADD_Wmask) ) & ~MXADD_Wmask;
        uint64_t result=x_result | y_result | z_result | w_result;
        return result;
    }
    
    template<int di, int dj, int dk>
    inline static uint64_t Packed_Offset(const uint64_t pI)
    {
        static const uint64_t pJ = LinearOffset<di,dj,dk>::value;

        uint64_t x_result=( (pI | MXADD_Xmask) + (pJ & ~MXADD_Xmask) ) & ~MXADD_Xmask;
        uint64_t y_result=( (pI | MXADD_Ymask) + (pJ & ~MXADD_Ymask) ) & ~MXADD_Ymask;
        uint64_t z_result=( (pI | MXADD_Zmask) + (pJ & ~MXADD_Zmask) ) & ~MXADD_Zmask;
        
        uint64_t result=x_result | y_result | z_result;
        return result;
    }

    template<int di>
    inline static uint64_t Packed_OffsetXdim(const uint64_t pI)
    {
        static const uint64_t pJ = LinearOffset<di,0,0>::value;

        uint64_t x_result=( (pI | MXADD_Xmask) + (pJ & ~MXADD_Xmask) ) & ~MXADD_Xmask;
        uint64_t not_x_result= pI & MXADD_Xmask;
        
        uint64_t result=x_result | not_x_result;
        return result;
    }

    template<int dj>
    inline static uint64_t Packed_OffsetYdim(const uint64_t pI)
    {
        static const uint64_t pJ = LinearOffset<0,dj,0>::value;

        uint64_t y_result=( (pI | MXADD_Ymask) + (pJ & ~MXADD_Ymask) ) & ~MXADD_Ymask;
        uint64_t not_y_result= pI & MXADD_Ymask;
        
        uint64_t result=y_result | not_y_result;
        return result;
    }
    
    template<int dk>
    inline static uint64_t Packed_OffsetZdim(const uint64_t pI)
    {
        static const uint64_t pJ = LinearOffset<0,0,dk>::value;

        uint64_t z_result=( (pI | MXADD_Zmask) + (pJ & ~MXADD_Zmask) ) & ~MXADD_Zmask;
        uint64_t not_z_result= pI & MXADD_Zmask;
        
        uint64_t result=z_result | not_z_result;
        return result;
    }

};

//#####################################################################
// Class SPGrid_Mask_base (2D)
//#####################################################################

template<int log2_struct>
class SPGrid_Mask_base<log2_struct,2>
{
protected:

    enum {
        data_bits=log2_struct,                     // Bits needed for indexing individual bytes within type T
        block_bits=12-data_bits,                   // Bits needed for indexing data elements within a block
        block_ybits=block_bits/2+(block_bits%2>0), // Bits needed for the z-coordinate of a data elements within a block
        block_xbits=block_bits/2                   // Bits needed for the x-coordinate of a data elements within a block
    };

    enum : uint64_t { // Bit masks for the upper 52 bits of memory addresses (page indices)
        page_ymask=(0x5555555555555555UL<<(2-block_bits%2))&0xfffffffffffff000UL,
        page_xmask=(0xaaaaaaaaaaaaaaaaUL<<(2-block_bits%2))&0xfffffffffffff000UL
    };
    
public:

    enum : uint32_t {elements_per_block=1u<<block_bits};    
};

//#####################################################################
// Class SPGrid_Mask (2D)
//#####################################################################

template<int log2_struct,int log2_field>
class SPGrid_Mask<log2_struct,log2_field,2>: public SPGrid_Mask_base<log2_struct,2>
{
public:
    enum {dim=2};
    enum {field_size=1<<log2_field};
    typedef SPGrid_Mask_base<log2_struct,dim> T_Mask_base;
    using T_Mask_base::data_bits;using T_Mask_base::block_bits;
    using T_Mask_base::block_xbits;using T_Mask_base::block_ybits;
    using T_Mask_base::page_xmask;using T_Mask_base::page_ymask;

    enum { // Bit masks for the lower 12 bits of memory addresses (element indices within a page)
        element_ymask=((1<<block_ybits)-1)<<log2_field,
        element_xmask=((1<<block_xbits)-1)<<(log2_field+block_ybits)
    };
    
    enum { 
        // Same as the corresponding element bit masks, but with the most significant bit the respective coordinate zeroed out
        element_y_lsbits=(element_ymask>>1)&element_ymask,
        element_x_lsbits=(element_xmask>>1)&element_xmask,
        
        // Same as the corresponding element bit masks, but with the least significant bit the respective coordinate zeroed out
        element_y_msbits=(element_ymask<<1)&element_ymask,
        element_x_msbits=(element_xmask<<1)&element_xmask,

        // Just the most significant bit of the element bit mask for the respective coordinate
        element_y_msbit=element_ymask^element_y_lsbits,
        element_x_msbit=element_xmask^element_x_lsbits,
        
        downsample_lower_mask =  element_y_lsbits | element_x_lsbits,
        upsample_lower_mask   =  element_y_msbits | element_x_msbits,

        lob = block_bits%2,
        
        xloc = lob==0 ? 13 : 12,
        yloc = lob==0 ? 12 : 13,

        u_ybit_shift = yloc - (log2_field+block_ybits-1),
        u_xbit_shift = xloc - (log2_field+block_bits-1),
        
        bit12_mask = lob==0 ? element_y_msbit : element_x_msbit,
        bit13_mask = lob==0 ? element_x_msbit : element_y_msbit,
    };

public:
    enum { // Bit masks for aggregate addresses
        ymask=page_ymask|(uint64_t)element_ymask,
        xmask=page_xmask|(uint64_t)element_xmask
    };
    enum { 
        MXADD_Ymask=~ymask, 
        MXADD_Xmask=~xmask, 
        MXADD_Wmask=xmask|ymask
    };
    enum {
        ODD_BITS=BitSpread<1,xmask>::value | BitSpread<1,ymask>::value
    };

public:

    static unsigned int Bytes_Per_Element()
    {return 1u<<data_bits;}

    static unsigned int Elements_Per_Block()
    {return 1u<<block_bits;}

    template<int i, int j> struct LinearOffset
    {
      static const uint64_t value = BitSpread<(uint64_t)i,xmask>::value | BitSpread<(uint64_t)j,ymask>::value;
    };

    inline static uint64_t Linear_Offset(const int i, const int j)
    {
#ifdef HASWELL
        return Bit_Spread(i,xmask)|Bit_Spread(j,ymask);
#else
        return Bit_Spread<xmask>(i)|Bit_Spread<ymask>(j);
#endif
    }
    
    inline static uint64_t Linear_Offset(const std::array<int,2>& coord)
    {
#ifdef HASWELL
        return Bit_Spread(coord[0],xmask)|Bit_Spread(coord[1],ymask);
#else
        return Bit_Spread<xmask>(coord[0])|Bit_Spread<ymask>(coord[1]);
#endif
    }
    inline static uint64_t Linear_Offset(const std::array<unsigned int,2>& coord)
    {
#ifdef HASWELL
        return Bit_Spread(coord[0],xmask)|Bit_Spread(coord[1],ymask);
#else
        return Bit_Spread<xmask>(coord[0])|Bit_Spread<ymask>(coord[1]);
#endif
    }

    inline static void LinearToCoord(uint64_t linear_offset, int* i, int* j)
    {
        *i = Bit_Pack(linear_offset,xmask);
        *j = Bit_Pack(linear_offset,ymask);
    }

    inline static std::array<int,2> LinearToCoord(uint64_t linear_offset)
    {
        std::array<int,2> coord;
        coord[0] = Bit_Pack(linear_offset,xmask);
        coord[1] = Bit_Pack(linear_offset,ymask);
        return coord;
    }

    // 1-based offset calculation
    inline static uint64_t DownsampleOffset(uint64_t linear_offset)
    {
        return Packed_RightShift(Packed_Offset<1,1>(linear_offset));
    }


    inline static uint64_t Packed_RightShift(uint64_t linear_offset)
    {
        static uint64_t my_array[4] = {
            0,
            bit12_mask,
            bit13_mask,
            bit12_mask|bit13_mask};
        uint64_t upper = (linear_offset >> 2) & 0xfffffffffffff000UL;
        uint64_t lower = (linear_offset >> 1) & downsample_lower_mask;
        uint64_t result = upper | lower | my_array[linear_offset>>12 & 0x3UL];
        return result;
    }
    
    // 1-based offset calculation
    inline static uint64_t UpsampleOffset(uint64_t linear_offset)
    {
        return Packed_Offset<-1,-1>(Packed_LeftShift(linear_offset));
    }

    inline static uint64_t Packed_LeftShift(uint64_t linear_offset)
    {
        uint64_t upper = (linear_offset << 2) & 0xfffffffffffff000UL;
        uint64_t lower = (linear_offset << 1) & upsample_lower_mask;

        uint64_t x_bit_place = (linear_offset & element_x_msbit)<<u_xbit_shift;
        uint64_t y_bit_place = (linear_offset & element_y_msbit)<<u_ybit_shift;

        uint64_t result = upper | lower | x_bit_place | y_bit_place;
        return result;
    }

    inline static uint64_t Packed_Add(const uint64_t i,const uint64_t j)
    {
        uint64_t x_result=( (i | MXADD_Xmask) + (j & ~MXADD_Xmask) ) & ~MXADD_Xmask;
        uint64_t y_result=( (i | MXADD_Ymask) + (j & ~MXADD_Ymask) ) & ~MXADD_Ymask;
        uint64_t w_result=( (i | MXADD_Wmask) + (j & ~MXADD_Wmask) ) & ~MXADD_Wmask;
        uint64_t result=x_result | y_result | w_result;
        return result;
    }

    template<uint64_t j>
    inline static uint64_t Packed_Add_Imm(const uint64_t i)
    {
        uint64_t x_result=( (i | MXADD_Xmask) + (j & ~MXADD_Xmask) ) & ~MXADD_Xmask;
        uint64_t y_result=( (i | MXADD_Ymask) + (j & ~MXADD_Ymask) ) & ~MXADD_Ymask;
        uint64_t w_result=( (i | MXADD_Wmask) + (j & ~MXADD_Wmask) ) & ~MXADD_Wmask;
        uint64_t result=x_result | y_result | w_result;
        return result;
    }
    
    template<int di, int dj>
    inline static uint64_t Packed_Offset(const uint64_t pI)
    {
        static const uint64_t pJ = LinearOffset<di,dj>::value;

        uint64_t x_result=( (pI | MXADD_Xmask) + (pJ & ~MXADD_Xmask) ) & ~MXADD_Xmask;
        uint64_t y_result=( (pI | MXADD_Ymask) + (pJ & ~MXADD_Ymask) ) & ~MXADD_Ymask;
        
        uint64_t result=x_result | y_result;
        return result;
    }

    template<int di>
    inline static uint64_t Packed_OffsetXdim(const uint64_t pI)
    {
        static const uint64_t pJ = LinearOffset<di,0>::value;

        uint64_t x_result=( (pI | MXADD_Xmask) + (pJ & ~MXADD_Xmask) ) & ~MXADD_Xmask;
        uint64_t not_x_result= pI & MXADD_Xmask;
        
        uint64_t result=x_result | not_x_result;
        return result;
    }

    template<int dj>
    inline static uint64_t Packed_OffsetYdim(const uint64_t pI)
    {
        static const uint64_t pJ = LinearOffset<0,dj>::value;

        uint64_t y_result=( (pI | MXADD_Ymask) + (pJ & ~MXADD_Ymask) ) & ~MXADD_Ymask;
        uint64_t not_y_result= pI & MXADD_Ymask;
        
        uint64_t result=y_result | not_y_result;
        return result;
    }

};
}
#endif
