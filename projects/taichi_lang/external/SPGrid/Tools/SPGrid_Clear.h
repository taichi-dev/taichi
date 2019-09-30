//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Clear
//#####################################################################
#ifndef __SPGrid_Clear_h__
#define __SPGrid_Clear_h__

#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Tools/SPGrid_Block_Iterator.h>

namespace SPGrid_Computations{

using namespace SPGrid;

template<class T_STRUCT,class T_FIELD,int d,int length=1> class Clear;

template<class T_STRUCT,class T_FIELD,int d>
class Clear<T_STRUCT,T_FIELD,d,1>
{
    T_FIELD T_STRUCT::* data_field;
public:
    Clear(T_FIELD T_STRUCT::* data_field_input) :data_field(data_field_input) {}

    Clear(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* data_field_input) :data_field(data_field_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array_mask<T_FIELD> Data_array_mask;
    auto data=allocator.Get_Array(data_field);
    for(SPGrid_Block_Iterator<Data_array_mask> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(data)=T_FIELD();}
};

template<class T_STRUCT,class T_FIELD,int d>
class Clear<T_STRUCT,T_FIELD,d,2>
{
    T_FIELD T_STRUCT::* data_field1;
    T_FIELD T_STRUCT::* data_field2;
public:
    Clear(T_FIELD T_STRUCT::* data_field1_input,T_FIELD T_STRUCT::* data_field2_input)
        :data_field1(data_field1_input),data_field2(data_field2_input) {}

    Clear(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* data_field1_input,T_FIELD T_STRUCT::* data_field2_input)
        :data_field1(data_field1_input),data_field2(data_field2_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
    Data_array_type data1=allocator.Get_Array(data_field1);
    Data_array_type data2=allocator.Get_Array(data_field2);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next()){
        iterator.Data(data1)=T_FIELD();
        iterator.Data(data2)=T_FIELD();}}
};

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d> void
Masked_Clear(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
    T_DATA T_STRUCT::* data_field,T_FLAGS T_STRUCT::* flags_field,const T_FLAGS mask)  
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;

    Data_array_type data=allocator.Get_Array(data_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);
    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(data)=T_DATA();
}

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d> void
Masked_Clear(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
    T_DATA T_STRUCT::* data1_field,T_DATA T_STRUCT::* data2_field,T_FLAGS T_STRUCT::* flags_field,const T_FLAGS mask)  
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;

    Data_array_type data1=allocator.Get_Array(data1_field);
    Data_array_type data2=allocator.Get_Array(data2_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);
    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask){
            iterator.Data(data1)=T_DATA();
            iterator.Data(data2)=T_DATA();}
}

//#####################################################################
}
#endif
