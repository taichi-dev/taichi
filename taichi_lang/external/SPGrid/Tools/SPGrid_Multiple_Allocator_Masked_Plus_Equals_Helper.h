//#####################################################################
// Copyright 2014, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Multiple_Allocator_Masked_Plus_Equals_Helper
//#####################################################################
#ifndef __SPGrid_Multiple_Allocator_Plus_Equals_Helper_h__
#define __SPGrid_Multiple_Allocator_Plus_Equals_Helper_h__

#include <SPGrid/Core/SPGrid_Allocator.h>
#include <Threading_Tools/PTHREAD_QUEUE.h>

extern PTHREAD_QUEUE* pthread_queue;

namespace SPGrid_Computations{
using namespace SPGrid;
template<class T_STRUCT,class T_DATA,class T_FLAGS,int d> class Multiple_Allocator_Masked_Plus_Equals_Helper;
//#####################################################################
// Multiple_Allocator_Masked_Plus_Equals_Threading_Operation_Helper
//#####################################################################
template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
struct Multiple_Allocator_Masked_Plus_Equals_Threading_Operation_Helper:public PTHREAD_QUEUE::TASK
{
    typedef std::pair<const unsigned long*,unsigned> T_BLOCK;

    const T_BLOCK blocks;
    Multiple_Allocator_Masked_Plus_Equals_Helper<T_STRUCT,T_DATA,T_FLAGS,d>* const obj;
    
    Multiple_Allocator_Masked_Plus_Equals_Threading_Operation_Helper(const T_BLOCK& blocks_input,Multiple_Allocator_Masked_Plus_Equals_Helper<T_STRUCT,T_DATA,T_FLAGS,d>* const obj_input)
        :blocks(blocks_input),obj(obj_input)
    {}
    void Run(){obj->Operation(blocks);}
};
//#####################################################################
// Class Multiple_Allocator_Masked_Plus_Equals_Helper
//#####################################################################
template<class T_STRUCT,class T_DATA,class T_FLAGS,int d>
class Multiple_Allocator_Masked_Plus_Equals_Helper
{
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_DATA>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T_DATA>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T_FLAGS>::type Const_flag_array_type;
    typedef std::pair<const unsigned long*,unsigned> T_BLOCK;

    SPGrid_Allocator<T_STRUCT,d>& source1_allocator;
    SPGrid_Allocator<T_STRUCT,d>& source2_allocator;
    SPGrid_Allocator<T_STRUCT,d>& destination_allocator;
    SPGrid_Allocator<T_STRUCT,d>& flags_allocator;
    const T_BLOCK blocks;
    T_DATA T_STRUCT::* source1_field;
    T_DATA T_STRUCT::* source2_field;
    T_DATA T_STRUCT::* destination_field;
    T_FLAGS T_STRUCT::* flags_field;
    const unsigned long mask;
//#####################################################################
public:
    Multiple_Allocator_Masked_Plus_Equals_Helper(SPGrid_Allocator<T_STRUCT,d>& source1_allocator_input,SPGrid_Allocator<T_STRUCT,d>& source2_allocator_input,SPGrid_Allocator<T_STRUCT,d>& destination_allocator_input,
        SPGrid_Allocator<T_STRUCT,d>& flags_allocator_input,const std::pair<const unsigned long*,unsigned>& blocks_input,T_DATA T_STRUCT::* source1_field_input,T_DATA T_STRUCT::* source2_field_input,
        T_DATA T_STRUCT::* destination_field_input,T_FLAGS T_STRUCT::* flags_field_input,const unsigned long mask_input)
        :source1_allocator(source1_allocator_input),source2_allocator(source2_allocator_input),destination_allocator(destination_allocator_input),flags_allocator(flags_allocator_input),blocks(blocks_input),
         source1_field(source1_field_input),source2_field(source2_field_input),destination_field(destination_field_input),flags_field(flags_field_input),mask(mask_input)
    {}
    
    void Run()
    {Operation(blocks);}

    void Operation(const std::pair<const unsigned long*,unsigned>& blocks_in) const
    {Const_data_array_type source1=source1_allocator.Get_Const_Array(source1_field);
    Const_data_array_type source2=source2_allocator.Get_Const_Array(source2_field);
    Data_array_type destination=destination_allocator.Get_Array(destination_field);
    Const_flag_array_type flags=flags_allocator.Get_Const_Array(flags_field);    
    for(SPGrid_Block_Iterator<typename Data_array_type::MASK> iterator(blocks_in);iterator.Valid();iterator.Next())
        if(iterator.Data(flags)&mask) iterator.Data(destination)=iterator.Data(source1)+iterator.Data(source2);}

    void Run_Parallel(const int number_of_partitions)
    {const unsigned long* block_offsets=blocks.first;
    const int size=blocks.second;
    if(size<number_of_partitions*16){Operation(blocks);return;}
    for(int partition=0;partition<number_of_partitions;partition++){
        int first_index_of_partition=(size/number_of_partitions)*(partition)+std::min(size%number_of_partitions,partition);
        int block_size=(size/number_of_partitions)+((partition<size%number_of_partitions)?1:0);
        T_BLOCK block(block_offsets+first_index_of_partition,block_size);
        Multiple_Allocator_Masked_Plus_Equals_Threading_Operation_Helper<T_STRUCT,T_DATA,T_FLAGS,d>* task=
            new Multiple_Allocator_Masked_Plus_Equals_Threading_Operation_Helper<T_STRUCT,T_DATA,T_FLAGS,d>(block,this);
        pthread_queue->Queue(task);}
    pthread_queue->Wait();}
//#####################################################################
};
}
#endif
