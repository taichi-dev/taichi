//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Masked_Average
//#####################################################################
#ifndef __SPGrid_Masked_Average_h__
#define __SPGrid_Masked_Average_h__

namespace SPGrid_Computations{

template<int stencil_size,class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Average
{
    T_DATA T_STRUCT::* source_field;
    T_DATA T_STRUCT::* destination_field;
    T_DATA T_STRUCT::* weights_field;
    T_FLAGS T_STRUCT::* flags_field;
    unsigned long (&stencil)[stencil_size];
    const T_FLAGS mask;
    const T_DATA weight;
 
public:
    Masked_Average(T_DATA T_STRUCT::* source_field_input,T_DATA T_STRUCT::* destination_field_input,T_DATA T_STRUCT::* weights_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,unsigned long (&stencil_input)[stencil_size],const T_FLAGS mask_input,const T_DATA weight_input)
        :source_field(source_field_input),destination_field(destination_field_input),weights_field(weights_field_input),
         flags_field(flags_field_input),stencil(stencil_input),mask(mask_input),weight(weight_input)
    {}

    Masked_Average(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_DATA T_STRUCT::* source_field_input,T_DATA T_STRUCT::* destination_field_input,T_DATA T_STRUCT::* weights_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,unsigned long (&stencil_input)[stencil_size],const T_FLAGS mask_input,const T_DATA weight_input)
        :source_field(source_field_input),destination_field(destination_field_input),weights_field(weights_field_input),
         flags_field(flags_field_input),stencil(stencil_input),mask(mask_input),weight(weight_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;        
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    Const_data_array_type source=allocator.Get_Const_Array(source_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Data_array_type weights=allocator.Get_Array(weights_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);
    for(SPGrid_Block_Iterator<typename Const_flag_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags) & mask)
            for(int i=0;i<stencil_size;i++){
                iterator.Data(destination,stencil[i])+=iterator.Data(source)*weight;
                iterator.Data(weights,stencil[i])+=weight;}}
};

//#####################################################################
}
#endif
