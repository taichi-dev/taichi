//#####################################################################
// Copyright 2010, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#ifndef __Shadow_Grid_Helper__
#define __Shadow_Grid_Helper__

#include <SPGrid/Core/SPGrid_Utilities.h>
#include <SPGrid/Core/SPGrid_Mask.h>

namespace SPGrid{
template<class T,int log2_struct, int d> class Shadow_Grid_Helper;

template<class T,int log2_struct>
class Shadow_Grid_Helper<T,log2_struct,2>
{
    enum{d=2};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T)>::value,d> T_MASK;

    enum {
        prefetch_degree = 0,
        block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        og_xsize = block_xsize+2,
        og_ysize = block_ysize+2,
        xmin = 1,
        ymin = 1,
        // Inclusive!!! give mins and maxs for actual block within shadow grid
        xmax = og_xsize-2,
        ymax = og_ysize-2,
    };

public:
    static void ComputeShadowGrid(unsigned long* offset_grid_ptr,unsigned long packed_offset);
};

template<class T,int log2_struct>
class Shadow_Grid_Helper<T,log2_struct,3>
{
    enum{d=3};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T)>::value,d> T_MASK;

    enum {
        prefetch_degree = 0,
        block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        block_zsize = 1u << T_MASK::block_zbits,
        og_xsize = block_xsize+2,
        og_ysize = block_ysize+2,
        og_zsize = block_zsize+2,
        xmin = 1,
        ymin = 1,
        zmin = 1,
        // Inclusive!!! give mins and maxs for actual block within shadow grid
        xmax = og_xsize-2,
        ymax = og_ysize-2,
        zmax = og_zsize-2
    };

public:
    static void ComputeShadowGrid(unsigned long* offset_grid_ptr,unsigned long packed_offset);
};
}
#endif
