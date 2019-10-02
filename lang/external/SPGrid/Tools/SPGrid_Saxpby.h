//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Saxpby
//#####################################################################
#ifndef __SPGrid_Saxpby_h__
#define __SPGrid_Saxpby_h__

#include <SPGrid/Core/SPGrid_Allocator.h>

namespace SPGrid_Computations{

using namespace SPGrid;

template<class T_STRUCT,class T_FIELD,int d> void
Saxpby(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
    T_FIELD T_STRUCT::* x_field,T_FIELD T_STRUCT::* y_field,const T_FIELD alpha,const T_FIELD beta,T_FIELD T_STRUCT::* destination_field)
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;

    Const_data_array_type x=allocator.Get_Const_Array(x_field);
    Const_data_array_type y=allocator.Get_Const_Array(y_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(destination)=alpha*iterator.Data(x)+beta*iterator.Data(y);
}

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d> void
Masked_Saxpby(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
    T_DATA T_STRUCT::* x_field,T_DATA T_STRUCT::* y_field,const T_DATA alpha,const T_DATA beta,T_DATA T_STRUCT::* destination_field,
    T_FLAGS T_STRUCT::* flags_field,const T_FLAGS mask)  
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;

    Const_data_array_type x=allocator.Get_Const_Array(x_field);
    Const_data_array_type y=allocator.Get_Const_Array(y_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);
    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(destination)=alpha*iterator.Data(x)+beta*iterator.Data(y);
}
//#####################################################################
}
#endif
