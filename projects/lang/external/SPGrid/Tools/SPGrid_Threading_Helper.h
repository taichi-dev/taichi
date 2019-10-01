//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Threading_Helper
//#####################################################################
#ifndef __SPGrid_Threading_Helper_h__
#define __SPGrid_Threading_Helper_h__

#include <vector>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <Threading_Tools/PTHREAD_QUEUE.h>

extern PTHREAD_QUEUE* pthread_queue;

namespace SPGrid_Computations{

using namespace SPGrid;

template<class T_STRUCT,int d,class T_OPERATION>
struct Threading_Operation_Helper:public PTHREAD_QUEUE::TASK
{
    typedef std::pair<const unsigned long*,unsigned> T_BLOCK;

    SPGrid_Allocator<T_STRUCT,d>& allocator;
    const T_BLOCK blocks;
    const T_OPERATION& operation;

    Threading_Operation_Helper(SPGrid_Allocator<T_STRUCT,d>& allocator_input,const T_BLOCK& blocks_input,const T_OPERATION& operation_input)
        :allocator(allocator_input),blocks(blocks_input),operation(operation_input) {}

    void Run(){operation.Run(allocator,blocks);}
};

template<class T_STRUCT,int d>
class Threading_Helper
{
    typedef std::pair<const unsigned long*,unsigned> T_BLOCK;
    
    SPGrid_Allocator<T_STRUCT,d>& allocator;
    const T_BLOCK& blocks;

public:
    Threading_Helper(SPGrid_Allocator<T_STRUCT,d>& allocator_input,const T_BLOCK& blocks_input)
        :allocator(allocator_input),blocks(blocks_input)
    {}

    template<class T_OPERATION>
    void Run_Parallel(const T_OPERATION& operation,std::vector<T_BLOCK> list_of_partitions)
    {for(int partition=0;partition<list_of_partitions.size();partition++){     
     Threading_Operation_Helper<T_STRUCT,d,T_OPERATION>* task=
         new Threading_Operation_Helper<T_STRUCT,d,T_OPERATION>(allocator,list_of_partitions[partition],operation);     
     pthread_queue->Queue(task);}
     pthread_queue->Wait();}

    template<class T_OPERATION>
    void Run_Parallel(const T_OPERATION& operation,const int number_of_partitions)
    {const unsigned long* block_offsets=blocks.first;
    const int size=blocks.second;
    if(size<number_of_partitions*16){operation.Run(allocator,blocks);return;}
    for(int partition=0;partition<number_of_partitions;partition++){
        int first_index_of_partition=(size/number_of_partitions)*(partition)+std::min(size%number_of_partitions,partition);
        int block_size=(size/number_of_partitions)+((partition<size%number_of_partitions)?1:0);
        T_BLOCK block(block_offsets+first_index_of_partition,block_size);
        Threading_Operation_Helper<T_STRUCT,d,T_OPERATION>* task=
            new Threading_Operation_Helper<T_STRUCT,d,T_OPERATION>(allocator,block,operation);
        pthread_queue->Queue(task);}
    pthread_queue->Wait();}
};

template<class Functor>
void Run_Parallel_Blocks(const std::pair<const uint64_t*,uint32_t>& blocks,Functor functor)
{
#pragma omp parallel for
    for(int b=0;b<blocks.second;b++)
        functor(blocks.first[b]);
}

//#####################################################################
}
#endif
