//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class SPGrid_Computations::Masked_Copy_Or_Clear
//#####################################################################
#ifndef __SPGrid_Masked_Copy_Or_Clear_h__
#define __SPGrid_Masked_Copy_Or_Clear_h__

#include <SPGrid/Core/SPGrid_Allocator.h>

namespace SPGrid_Computations{

using namespace SPGrid;

template<class T_STRUCT,class T_FIELD,class T_FLAGS,int d>
class Masked_Copy_Or_Clear
{
    T_FIELD T_STRUCT::* data1_field;
    T_FIELD T_STRUCT::* data2_field;
    T_FLAGS T_STRUCT::* flags_field;
    const T_FLAGS mask1;
    const T_FLAGS mask2;
    
public:
    Masked_Copy_Or_Clear(T_FIELD T_STRUCT::* data1_field_input,T_FIELD T_STRUCT::* data2_field_input,T_FLAGS T_STRUCT::* flags_field_input,
        const T_FLAGS mask1_input,const T_FLAGS mask2_input)
        :data1_field(data1_field_input),data2_field(data2_field_input),flags_field(flags_field_input),mask1(mask1_input),mask2(mask2_input) {}

    Masked_Copy_Or_Clear(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* data1_field_input,T_FIELD T_STRUCT::* data2_field_input,T_FLAGS T_STRUCT::* flags_field_input,
        const T_FLAGS mask1_input,const T_FLAGS mask2_input)
        :data1_field(data1_field_input),data2_field(data2_field_input),flags_field(flags_field_input),mask1(mask1_input),mask2(mask2_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {
        typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
        typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
        typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;

        Data_array_type data1=allocator.Get_Array(data1_field);
        Const_data_array_type data2=allocator.Get_Const_Array(data2_field);
        Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);
        
        for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
            if(iterator.Data(flags) & mask1)
                iterator.Data(data1) = iterator.Data(data2);
            else if(iterator.Data(flags) & mask2)
                iterator.Data(data1) = T_FIELD();
    }

//#####################################################################
};
}
#endif
