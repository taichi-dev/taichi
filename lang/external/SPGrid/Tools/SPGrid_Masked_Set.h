//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Masked_Set
//#####################################################################
#ifndef __SPGrid_Masked_Set_h__
#define __SPGrid_Masked_Set_h__

namespace SPGrid_Computations{

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d> void
Masked_Set(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
    T_DATA T_STRUCT::* data_field,T_FLAGS T_STRUCT::* flags_field,const T_FLAGS mask,const T_DATA value)
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;

    Data_array_type data=allocator.Get_Array(data_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);

    for(SPGrid_Block_Iterator<typename Const_flag_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags) & mask)
            iterator.Data(data)=value;
}

//#####################################################################
}
#endif
