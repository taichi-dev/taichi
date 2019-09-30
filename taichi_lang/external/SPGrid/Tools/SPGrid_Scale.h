//#####################################################################
// Scaleright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Scale
//#####################################################################
#ifndef __SPGrid_Scale_h__
#define __SPGrid_Scale_h__

#include <SPGrid/Core/SPGrid_Allocator.h>

namespace SPGrid_Computations{

using namespace SPGrid;

template<class T_STRUCT,class T_FIELD,int d> void
Scale(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
      T_FIELD T_STRUCT::* field,const T_FIELD scale)
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;

    Data_array_type data=allocator.Get_Array(field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(data)*=scale;
}

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d> void
Masked_Scale(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
      T_DATA T_STRUCT::* field,const T_DATA scale,T_FLAGS T_STRUCT::* flags_field,const T_FLAGS mask)  
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    
    Data_array_type data=allocator.Get_Array(field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);
    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(data)*=scale;
}
//#####################################################################
}
#endif
