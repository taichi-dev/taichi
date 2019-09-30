//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Downsample_Accumulate_Shared
//#####################################################################
#ifndef __SPGrid_Downsample_Accumulate_Shared_h__
#define __SPGrid_Downsample_Accumulate_Shared_h__

namespace SPGrid_Computations{

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d> void
Downsample_Accumulate_Shared(SPGrid_Allocator<T_STRUCT,d>& allocator,SPGrid_Allocator<T_STRUCT,d>& coarse_allocator,
    const std::pair<const unsigned long*,unsigned>& blocks,T_DATA T_STRUCT::* data_field,T_DATA T_STRUCT::* weights_field,
    T_FLAGS T_STRUCT::* flags_field,const T_FLAGS mask)
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;

    Const_data_array_type data=allocator.Get_Const_Array(data_field);
    Const_data_array_type weights=allocator.Get_Const_Array(weights_field);
    Data_array_type coarse_data=coarse_allocator.Get_Array(data_field);
    Data_array_type coarse_weights=coarse_allocator.Get_Array(weights_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);

    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags) & mask){
            // Note: the iterator must be templatized over the *data* mask, for the following line to work properly
            unsigned long coarse_offset=Data_array_type::MASK::DownsampleOffset(iterator.Offset());
            coarse_data(coarse_offset)+=iterator.Data(data);
            coarse_weights(coarse_offset)+=iterator.Data(weights);}
}

//#####################################################################
}
#endif
