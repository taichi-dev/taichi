//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Masked_Average_Offset_Grid
//#####################################################################
#ifndef __SPGrid_Masked_Average_Offset_Grid_h__
#define __SPGrid_Masked_Average_Offset_Grid_h__

#include <SPGrid/Tools/Shadow_Grid_Helper.h>
#include <stdlib.h>  // malloc and free

namespace SPGrid_Computations{

// Helper structs
template<int stencil_size,int log2_struct,class T_DATA,class T_FLAGS,int d> struct Masked_Average_Offset_Grid_Helper;

template<int stencil_size,int log2_struct,class T_DATA,class T_FLAGS>
struct Masked_Average_Offset_Grid_Helper<stencil_size,log2_struct,T_DATA,T_FLAGS,2>
{
    enum{d=2};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T_DATA)>::value,d> T_MASK;
    enum{block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        og_xsize = block_xsize+2,
        og_ysize = block_ysize+2,
        xmin = 1,
        ymin = 1,
        xmax = og_xsize-2,
        ymax = og_ysize-2};
    typedef unsigned long (&offset_grid_type)[og_xsize][og_ysize];

    static inline void Run_Block(const T_DATA* const source,const unsigned long destination_base_addr,const unsigned long weights_base_addr,
        const T_FLAGS* const flags,const T_DATA weight,const T_FLAGS mask,const unsigned long offset,const int (&stencil)[stencil_size])
    {
        unsigned long* offset_grid_ptr = (unsigned long*)malloc( (og_xsize) * (og_ysize) * sizeof(unsigned long));
        offset_grid_type o_grid = reinterpret_cast<offset_grid_type>(*offset_grid_ptr);
        Shadow_Grid_Helper<T_DATA,log2_struct,d>::ComputeShadowGrid(offset_grid_ptr,offset);
        int cur_index = 0;
        // Actually process elements
        for(int i=xmin;i<=xmax;i++)
        for(int j=ymin;j<=ymax;j++)
        {
            unsigned flag = flags[cur_index];
            if( flag & mask ){
                const T_DATA value=weight*source[cur_index];
                for(int s=0;s<stencil_size;s++){
                    (*reinterpret_cast<T_DATA*>(destination_base_addr + *(&o_grid[i][j]+stencil[s]) ))+=value;
                    (*reinterpret_cast<T_DATA*>(weights_base_addr     + *(&o_grid[i][j]+stencil[s]) ))+=weight;}
            }
            cur_index++;
        }
        free(offset_grid_ptr);
    }
};

template<int stencil_size,int log2_struct,class T_DATA,class T_FLAGS>
struct Masked_Average_Offset_Grid_Helper<stencil_size,log2_struct,T_DATA,T_FLAGS,3>
{
    enum{d=3};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T_DATA)>::value,d> T_MASK;
    enum{block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        block_zsize = 1u << T_MASK::block_zbits,
        og_xsize = block_xsize+2,
        og_ysize = block_ysize+2,
        og_zsize = block_zsize+2,
        xmin = 1,
        ymin = 1,
        zmin = 1,
        xmax = og_xsize-2,
        ymax = og_ysize-2,
        zmax = og_zsize-2};
    typedef unsigned long (&offset_grid_type)[og_xsize][og_ysize][og_zsize];

    static inline void Run_Block(const T_DATA* const source,const unsigned long destination_base_addr,const unsigned long weights_base_addr,
        const T_FLAGS* const flags,const T_DATA weight,const T_FLAGS mask,const unsigned long offset,const int (&stencil)[stencil_size])
    {
        unsigned long* offset_grid_ptr = (unsigned long*)malloc( (og_xsize) * (og_ysize) * (og_zsize) * sizeof(unsigned long));
        offset_grid_type o_grid = reinterpret_cast<offset_grid_type>(*offset_grid_ptr);
        Shadow_Grid_Helper<T_DATA,log2_struct,d>::ComputeShadowGrid(offset_grid_ptr,offset);
        int cur_index = 0;
        // Actually process elements
        for(int i=xmin;i<=xmax;i++)
        for(int j=ymin;j<=ymax;j++)
        for(int k=zmin;k<=zmax;k++)
        {
            unsigned flag = flags[cur_index];
            if( flag & mask ){
                const T_DATA value=weight*source[cur_index];
                for(int s=0;s<stencil_size;s++){
                    (*reinterpret_cast<T_DATA*>(destination_base_addr + *(&o_grid[i][j][k]+stencil[s]) ))+=value;
                    (*reinterpret_cast<T_DATA*>(weights_base_addr     + *(&o_grid[i][j][k]+stencil[s]) ))+=weight;}
            }
            cur_index++;
        }
        free(offset_grid_ptr);
    }
};

// Class
template<int stencil_size,class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Average_Offset_Grid
{
    T_DATA T_STRUCT::* source_field;
    T_DATA T_STRUCT::* destination_field;
    T_DATA T_STRUCT::* weights_field;
    T_FLAGS T_STRUCT::* flags_field;
    const int (&stencil)[stencil_size];
    const T_FLAGS mask;
    const T_DATA weight;
 
public:
    Masked_Average_Offset_Grid(T_DATA T_STRUCT::* source_field_input,T_DATA T_STRUCT::* destination_field_input,T_DATA T_STRUCT::* weights_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const int (&stencil_input)[stencil_size],const T_FLAGS mask_input,const T_DATA weight_input)
        :source_field(source_field_input),destination_field(destination_field_input),weights_field(weights_field_input),
         flags_field(flags_field_input),stencil(stencil_input),mask(mask_input),weight(weight_input)
    {}

    Masked_Average_Offset_Grid(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_DATA T_STRUCT::* source_field_input,T_DATA T_STRUCT::* destination_field_input,T_DATA T_STRUCT::* weights_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,int (&stencil_input)[stencil_size],const T_FLAGS mask_input,const T_DATA weight_input)
        :source_field(source_field_input),destination_field(destination_field_input),weights_field(weights_field_input),
         flags_field(flags_field_input),stencil(stencil_input),mask(mask_input),weight(weight_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::mask T_MASK;
    Const_data_array_type source=allocator.Get_Const_Array(source_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Data_array_type weights=allocator.Get_Array(weights_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);    
    unsigned long destination_base_addr = reinterpret_cast<unsigned long>(destination.Get_Data_Ptr());
    unsigned long weights_base_addr     = reinterpret_cast<unsigned long>(weights.Get_Data_Ptr());
    for(SPGrid_Block_Iterator<typename Const_flag_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next_Block())
        Masked_Average_Offset_Grid_Helper<stencil_size,NextLogTwo<sizeof(T_STRUCT)>::value,T_DATA,T_FLAGS,d>::Run_Block(&iterator.Data(source),destination_base_addr,
            weights_base_addr,&iterator.Data(flags),weight,mask,iterator.Offset(),stencil);}
};

//#####################################################################
}
#endif
