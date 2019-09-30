//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Copy
//#####################################################################
#ifndef __SPGrid_Copy_h__
#define __SPGrid_Copy_h__

#include <SPGrid/Core/SPGrid_Allocator.h>

namespace SPGrid_Computations{

using namespace SPGrid;

template<class T_STRUCT,class T_FIELD,int d>
class Copy
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
    T_FIELD T_STRUCT::* source_field;
    T_FIELD T_STRUCT::* destination_field;
public:
    Copy(T_FIELD T_STRUCT::* source_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source_field(source_field_input),destination_field(destination_field_input)
    {}

    Copy(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* source_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source_field(source_field_input),destination_field(destination_field_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source=allocator.Get_Const_Array(source_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(destination)=iterator.Data(source);}
};

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Copy
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    T_DATA T_STRUCT::* source_field;
    T_DATA T_STRUCT::* destination_field;
    T_FLAGS T_STRUCT::* flags_field;
    const unsigned long mask;
public:
    Masked_Copy(T_DATA T_STRUCT::* source_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source_field(source_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {}
    
    Masked_Copy(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_DATA T_STRUCT::* source_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source_field(source_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {Run(allocator,blocks);}
    
    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source=allocator.Get_Const_Array(source_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(destination)=iterator.Data(source);}
};
//#####################################################################
}
#endif
