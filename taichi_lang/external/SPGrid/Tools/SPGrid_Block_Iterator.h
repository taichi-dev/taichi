//#####################################################################
// Copyright 2012-2013, Sean Bauer, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Classes SPGrid_Block_Iterator/SPGrid_Reverse_Block_Iterator
//#####################################################################
#ifndef __SPGrid_Block_Iterator_h__
#define __SPGrid_Block_Iterator_h__

#include <array>
#include <utility>

namespace SPGrid{

template<class T_MASK>
class SPGrid_Block_Iterator
{
public:
    const unsigned long* const block_offsets;
    const unsigned size;
    int block_index;
    unsigned element_index;
    enum {elements_per_block = T_MASK::elements_per_block};
    enum {element_index_mask = elements_per_block-1};

    SPGrid_Block_Iterator(const unsigned long* const block_offsets_input,const unsigned size_input)
        :block_offsets(block_offsets_input),size(size_input),block_index(0),element_index(0) {}
    
    SPGrid_Block_Iterator(const std::pair<const unsigned long*,unsigned>& blocks)
        :block_offsets(blocks.first),size(blocks.second),block_index(0),element_index(0) {}
    
    inline bool Valid() const
    {return block_index<(int)size;}

    inline void Next()
    {element_index=(element_index+1)&element_index_mask;if(!element_index) block_index++;}

    inline void Next_Block()
    {element_index=0;block_index++;}

    template<class T_ARRAY> inline
    typename T_ARRAY::DATA& Data(T_ARRAY& array) const
    {
        typedef typename T_ARRAY::DATA T;
        unsigned long block_addr = reinterpret_cast<unsigned long>(array.Get_Data_Ptr()) + block_offsets[block_index];

        return reinterpret_cast<T*>(block_addr)[element_index];
    }

    template<class T_ARRAY> inline
    typename T_ARRAY::DATA& Data(void* data_ptr) const
    {
        typedef typename T_ARRAY::DATA T;
        unsigned long block_addr = reinterpret_cast<unsigned long>(data_ptr) + block_offsets[block_index];

        return reinterpret_cast<T*>(block_addr)[element_index];
    }

    template<class T_ARRAY> inline
    typename T_ARRAY::DATA& Data(T_ARRAY& array, unsigned long offset_input) const
    {
        typedef typename T_ARRAY::DATA T;
        unsigned long offset = T_ARRAY::MASK::Packed_Add(block_offsets[block_index]  + element_index*sizeof(T), offset_input);
        unsigned long data_addr = reinterpret_cast<unsigned long>(array.Get_Data_Ptr()) + offset;

        return *reinterpret_cast<T*>(data_addr);
    }

    template<class T_ARRAY> inline
    typename T_ARRAY::DATA& Data(void* data_ptr, unsigned long offset_input) const
    {
        typedef typename T_ARRAY::DATA T;
        unsigned long offset = T_ARRAY::MASK::Packed_Add(block_offsets[block_index]  + element_index*sizeof(T), offset_input);
        unsigned long data_addr = reinterpret_cast<unsigned long>(data_ptr) + offset;

        return *reinterpret_cast<T*>(data_addr);
    }

    template<class T_MASK2> inline unsigned long Offset() const
    {return block_offsets[block_index]+(unsigned long)(element_index*T_MASK2::field_size);}
     
    template<class T_MASK2> inline unsigned long Offset(unsigned long offset_input) const
    {return T_MASK2::Packed_Add(block_offsets[block_index]+(unsigned long)(element_index*T_MASK2::field_size),offset_input);}
    
    inline unsigned long Offset() const
    {return block_offsets[block_index]+(unsigned long)(element_index*T_MASK::field_size);}
    
    inline unsigned long Offset(unsigned long offset_input) const
    {return T_MASK::Packed_Add(block_offsets[block_index]+(unsigned long)(element_index*T_MASK::field_size),offset_input);}
    
    // NOTE: The following is expensive
    std::array<int,T_MASK::dim> Index() const
    {return T_MASK::LinearToCoord(block_offsets[block_index]+(unsigned long)(element_index*T_MASK::field_size));}
//#####################################################################
};

template<class T_MASK>
class SPGrid_Reverse_Block_Iterator
{
public:
    const unsigned long* const block_offsets;
    const unsigned size;
    int block_index;
    unsigned element_index; // Must be signed to detect <0
    enum {elements_per_block = T_MASK::elements_per_block};
    enum {element_index_mask = elements_per_block-1};

    SPGrid_Reverse_Block_Iterator(const unsigned long* const block_offsets_input,const unsigned size_input)
        :block_offsets(block_offsets_input),size(size_input),block_index(size-1),element_index(elements_per_block-1) {}

    SPGrid_Reverse_Block_Iterator(const std::pair<const unsigned long*,unsigned>& blocks)
        :block_offsets(blocks.first),size(blocks.second),block_index(size-1),element_index(elements_per_block-1) {}
    
    inline bool Valid() const
    {return block_index>=0;}

    inline void Next() 
    {if(!element_index) block_index--;element_index=(element_index+element_index_mask)&element_index_mask;}

    template<class T_ARRAY> inline
    typename T_ARRAY::DATA& Data(T_ARRAY& array) const
    {
        typedef typename T_ARRAY::DATA T;
        unsigned long block_addr = reinterpret_cast<unsigned long>(array.Get_Data_Ptr()) + block_offsets[block_index];

        return reinterpret_cast<T*>(block_addr)[element_index];
    }

    template<class T_ARRAY> inline
    typename T_ARRAY::DATA& Data(void* data_ptr) const
    {
        typedef typename T_ARRAY::DATA T;
        unsigned long block_addr = reinterpret_cast<unsigned long>(data_ptr) + block_offsets[block_index];

        return reinterpret_cast<T*>(block_addr)[element_index];
    }

    template<class T_ARRAY> inline
    typename T_ARRAY::DATA& Data(T_ARRAY& array, unsigned long offset_input) const
    {
        typedef typename T_ARRAY::DATA T;
        unsigned long offset = T_ARRAY::MASK::Packed_Add(block_offsets[block_index]  + element_index*sizeof(T), offset_input);
        unsigned long data_addr = reinterpret_cast<unsigned long>(array.Get_Data_Ptr()) + offset;

        return *reinterpret_cast<T*>(data_addr);
    }

    template<class T_ARRAY> inline
    typename T_ARRAY::DATA& Data(void* data_ptr, unsigned long offset_input) const
    {
        typedef typename T_ARRAY::DATA T;
        unsigned long offset = T_ARRAY::MASK::Packed_Add(block_offsets[block_index]  + element_index*sizeof(T), offset_input);
        unsigned long data_addr = reinterpret_cast<unsigned long>(data_ptr) + offset;

        return *reinterpret_cast<T*>(data_addr);
    }

    template<class T_MASK2> inline unsigned long Offset() const
    {return block_offsets[block_index]+(unsigned long)(element_index*T_MASK2::field_size);}
    
    inline unsigned long Offset() const
    {return block_offsets[block_index]+(unsigned long)(element_index*T_MASK::field_size);}
    
    // NOTE: The following is expensive
    std::array<int,T_MASK::dim> Index() const
    {return T_MASK::LinearToCoord(block_offsets[block_index]+(unsigned long)(element_index*T_MASK::field_size));}

//#####################################################################
};
}
#endif
