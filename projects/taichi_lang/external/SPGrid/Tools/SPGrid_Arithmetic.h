//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Add,Subtract,Multiply,Divide (componentwise)
//#####################################################################
#ifndef __SPGrid_Arithmetic_h__
#define __SPGrid_Arithmetic_h__

#include <SPGrid/Core/SPGrid_Allocator.h>

namespace SPGrid_Computations{

using namespace SPGrid;
//#####################################################################
template<class T_STRUCT,class T_FIELD,int d>
class Add
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
    T_FIELD T_STRUCT::* source1_field;
    T_FIELD T_STRUCT::* source2_field;
    T_FIELD T_STRUCT::* destination_field;
public:
    Add(T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {}

    Add(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(destination)=iterator.Data(source1)+iterator.Data(source2);}
};

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Add
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    T_DATA T_STRUCT::* source1_field;
    T_DATA T_STRUCT::* source2_field;
    T_DATA T_STRUCT::* destination_field;
    T_FLAGS T_STRUCT::* flags_field;
    const unsigned long mask;
public:
    Masked_Add(T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {}
    
    Masked_Add(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {Run(allocator,blocks);}
    
    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(destination)=iterator.Data(source1)+iterator.Data(source2);}
};
//#####################################################################
template<class T_STRUCT,class T_FIELD,int d>
class Subtract
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
    T_FIELD T_STRUCT::* source1_field;
    T_FIELD T_STRUCT::* source2_field;
    T_FIELD T_STRUCT::* destination_field;
public:
    Subtract(T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {}

    Subtract(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(destination)=iterator.Data(source1)-iterator.Data(source2);}
};

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Subtract
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    T_DATA T_STRUCT::* source1_field;
    T_DATA T_STRUCT::* source2_field;
    T_DATA T_STRUCT::* destination_field;
    T_FLAGS T_STRUCT::* flags_field;
    const unsigned long mask;
public:
    Masked_Subtract(T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {}
    
    Masked_Subtract(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {Run(allocator,blocks);}
    
    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(destination)=iterator.Data(source1)-iterator.Data(source2);}
};
//#####################################################################
template<class T_STRUCT,class T_FIELD,int d>
class Multiply
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
    T_FIELD T_STRUCT::* source1_field;
    T_FIELD T_STRUCT::* source2_field;
    T_FIELD T_STRUCT::* destination_field;
public:
    Multiply(T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {}

    Multiply(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(destination)=iterator.Data(source1)*iterator.Data(source2);}
};

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Multiply
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    T_DATA T_STRUCT::* source1_field;
    T_DATA T_STRUCT::* source2_field;
    T_DATA T_STRUCT::* destination_field;
    T_FLAGS T_STRUCT::* flags_field;
    const unsigned long mask;
public:
    Masked_Multiply(T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {}
    
    Masked_Multiply(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {Run(allocator,blocks);}
    
    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(destination)=iterator.Data(source1)*iterator.Data(source2);}
};
//#####################################################################
template<class T_STRUCT,class T_FIELD,int d>
class Divide
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
    T_FIELD T_STRUCT::* source1_field;
    T_FIELD T_STRUCT::* source2_field;
    T_FIELD T_STRUCT::* destination_field;
public:
    Divide(T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {}

    Divide(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(destination)=iterator.Data(source1)/iterator.Data(source2);}
};

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Divide
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    T_DATA T_STRUCT::* source1_field;
    T_DATA T_STRUCT::* source2_field;
    T_DATA T_STRUCT::* destination_field;
    T_FLAGS T_STRUCT::* flags_field;
    const unsigned long mask;
public:
    Masked_Divide(T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {}
    
    Masked_Divide(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {Run(allocator,blocks);}
    
    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(destination)=iterator.Data(source1)/iterator.Data(source2);}
};
//#####################################################################
template<class T_STRUCT,class T_FIELD,int d>
class Saxpy
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
    const T_FIELD alpha;
    T_FIELD T_STRUCT::* source1_field;
    T_FIELD T_STRUCT::* source2_field;
    T_FIELD T_STRUCT::* destination_field;
public:
    Saxpy(T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {}

    Saxpy(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* source1_field_input,T_FIELD T_STRUCT::* source2_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(destination)=alpha*iterator.Data(source1)+iterator.Data(source2);}
};

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Saxpy
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    const T_DATA alpha;
    T_DATA T_STRUCT::* source1_field;
    T_DATA T_STRUCT::* source2_field;
    T_DATA T_STRUCT::* destination_field;
    T_FLAGS T_STRUCT::* flags_field;
    const unsigned long mask;
public:
    Masked_Saxpy(const T_DATA alpha_input,T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :alpha(alpha_input),source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {}
    
    Masked_Saxpy(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        const T_DATA alpha_input,T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,T_DATA T_STRUCT::* destination_field_input,
        T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :alpha(alpha_input),source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),
         flags_field(flags_field_input),mask(mask_input)
    {Run(allocator,blocks);}
    
    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source1=allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=allocator.Get_Const_Array(source2_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask)
            iterator.Data(destination)=alpha*iterator.Data(source1)+iterator.Data(source2);}
};
//#####################################################################
template<class T_STRUCT,class T_FIELD,int d>
class Invert
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FIELD>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_FIELD>::type Data_array_type;
    T_FIELD T_STRUCT::* source_field;
    T_FIELD T_STRUCT::* destination_field;
public:
    Invert(T_FIELD T_STRUCT::* source_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source_field(source_field_input),destination_field(destination_field_input)
    {}

    Invert(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_FIELD T_STRUCT::* source_field_input,T_FIELD T_STRUCT::* destination_field_input)
        :source_field(source_field_input),destination_field(destination_field_input)
    {Run(allocator,blocks);}

    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source=allocator.Get_Const_Array(source_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        iterator.Data(destination)=(T_FIELD)1./(T_FIELD)iterator.Data(source);}
};

template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Masked_Invert
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    T_DATA T_STRUCT::* source_field;
    T_DATA T_STRUCT::* destination_field;
    T_FLAGS T_STRUCT::* flags_field;
    const unsigned long mask;
public:
    Masked_Invert(T_DATA T_STRUCT::* source_field_input,T_DATA T_STRUCT::* destination_field_input,T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source_field(source_field_input),destination_field(destination_field_input),flags_field(flags_field_input),mask(mask_input)
    {}
    
    Masked_Invert(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks,
        T_DATA T_STRUCT::* source_field_input,T_DATA T_STRUCT::* destination_field_input,T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source_field(source_field_input),destination_field(destination_field_input),flags_field(flags_field_input),mask(mask_input)
    {Run(allocator,blocks);}
    
    void Run(SPGrid_Allocator<T_STRUCT,d>& allocator,const std::pair<const unsigned long*,unsigned>& blocks) const
    {Const_data_array_type source=allocator.Get_Const_Array(source_field);
    Data_array_type destination=allocator.Get_Array(destination_field);
    Const_flag_array_type flags=allocator.Get_Const_Array(flags_field);    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask) iterator.Data(destination)=(T_DATA)1./(T_DATA)iterator.Data(source);}
};
//#####################################################################

}
#endif
