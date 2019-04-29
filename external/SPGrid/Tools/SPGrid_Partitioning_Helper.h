//#####################################################################
// Copyright 2013, Raj Setaluri, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Subroutine SPGrid_Computations::Partitioning_Helper
//#####################################################################
#ifndef __SPGrid_Partitioning_Helper_h__
#define __SPGrid_Partitioning_Helper_h__

#include <vector>
#include <algorithm>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Data_Structures/std_array.h>

namespace SPGrid_Computations{

using namespace SPGrid;

template<class T_STRUCT,int d> struct Partitioning_Helper_Neighbor_Offsets;
template<class T_STRUCT> struct Partitioning_Helper_Neighbor_Offsets<T_STRUCT,2>{
    enum{d=2};
    enum{num_neighbors=9};
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<unsigned>::mask T_MASK;
    static std::vector<unsigned long> Neighbor_Offsets(std_array<unsigned,d> block_size){
        std::vector<unsigned long> neighbors;neighbors.resize(num_neighbors);
        int count=0;
        for(int i=-1;i<=1;i++)
        for(int j=-1;j<=1;j++)
            neighbors[count++]=T_MASK::Linear_Offset(block_size*std_array<int,d>(i,j));
        return neighbors;
    }
};
template<class T_STRUCT> struct Partitioning_Helper_Neighbor_Offsets<T_STRUCT,3>{
    enum{d=3};
    enum{num_neighbors=27};
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<unsigned>::mask T_MASK;
    static std::vector<unsigned long> Neighbor_Offsets(std_array<unsigned,d> block_size){
        std::vector<unsigned long> neighbors;neighbors.resize(num_neighbors);
        int count=0;
        for(int i=-1;i<=1;i++)
        for(int j=-1;j<=1;j++)
        for(int k=-1;k<=1;k++)
            neighbors[count++]=T_MASK::Linear_Offset(block_size*std_array<int,d>(i,j,k));
        return neighbors;
    }
};

template<class T_STRUCT,int d>
class Partitioning_Helper
{    
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<unsigned>::mask T_MASK;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<unsigned>::type Flag_array_type;

    typedef std::pair<const unsigned long*,unsigned> T_BLOCK;
    typedef std::vector<T_BLOCK> TV_BLOCK;
    typedef std::pair<TV_BLOCK,TV_BLOCK> T_PARTITION;
    
    SPGrid_Allocator<T_STRUCT,d>& allocator;
    SPGrid_Set<Flag_array_type>& set;
    std::vector<unsigned long>& blocks;
    enum{ALL_ONES=0xFFFFFFFFu};

private:
    static bool lex_compare_packed_offsets(const unsigned long a,const unsigned long b)
    {const std_array<int,d> a_index(T_MASK::LinearToCoord(a));const std_array<int,d> b_index(T_MASK::LinearToCoord(b));
     for(int v=0;v<d;v++) if(a_index(v)!=b_index(v)) return a_index(v)<b_index(v);return true;}
    
public:
    Partitioning_Helper(SPGrid_Allocator<T_STRUCT,d>& allocator_input,SPGrid_Set<Flag_array_type>& set_input,std::vector<unsigned long>& blocks_input)
        :allocator(allocator_input),set(set_input),blocks(blocks_input)
    {}

    void Generate_Red_Black_Partition(const int number_of_partitions,TV_BLOCK& red_blocks,TV_BLOCK& black_blocks)
    {
        const int size=blocks.size();
        if(!size) return;
        // init neighbor offsets
        const std::vector<unsigned long> neighbor_offsets(Partitioning_Helper_Neighbor_Offsets<T_STRUCT,d>::Neighbor_Offsets(allocator.Block_Size()));        
        // sort new blocks *lexicographically*
        std::sort(blocks.begin(),blocks.end(),&Partitioning_Helper::lex_compare_packed_offsets);
        unsigned long minimum_offset=blocks[0];
        int current_index=0;
        for(int partition=0;partition<number_of_partitions;partition++){
            // find where this partition should end
            const int ideal_last_index_of_partition=(size/number_of_partitions)*(partition+1)+std::min(size%number_of_partitions,partition+1)-1;
            int last_index_of_partition=current_index;
            while((last_index_of_partition < (size-1)) &&
                  ((last_index_of_partition < ideal_last_index_of_partition) ||
                   (lex_compare_packed_offsets(blocks[last_index_of_partition],minimum_offset))))
                last_index_of_partition++;
            // create block
            T_BLOCK block(&blocks[0]+current_index,last_index_of_partition-current_index+1);
            // resort by spgrid index locally
            std::sort(blocks.begin()+current_index,blocks.begin()+last_index_of_partition+1);
            // update current_index and min_index
            current_index=last_index_of_partition+1;
            for(unsigned i=0;i<block.second;i++){
                const unsigned long block_offset=*(block.first+i);
                for(int j=0;j<=neighbor_offsets.size();j++){
                    unsigned long neighbor=T_MASK::Packed_Add(block_offset,neighbor_offsets[j]);
                    if(set.Is_Set(neighbor,ALL_ONES))
                        minimum_offset=std::max(minimum_offset,neighbor,&Partitioning_Helper::lex_compare_packed_offsets);}}
            // add block to red or black
            if(partition%2) black_blocks.push_back(block);
            else red_blocks.push_back(block);
            // dont go too far
            if(current_index >= size) break;}
    }
};
//#####################################################################
}
#endif
