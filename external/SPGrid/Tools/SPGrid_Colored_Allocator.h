//#####################################################################
// Copyright (c) 2018, Haixiang Liu, Eftychios Sifakis
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Colored_Allocator_h__
#define __SPGrid_Colored_Allocator_h__

#include <SPGrid/Core/SPGrid_Allocator.h>
#include <iostream>

namespace SPGrid{
//#####################################################################
// Class SPGrid_Colored_Allocator
//#####################################################################

template<class Struct_type,int dim,int log2_page=12>
class SPGrid_Colored_Allocator: public SPGrid_Allocator<Struct_type,dim,log2_page>
{
    using Base = SPGrid_Allocator<Struct_type,dim,log2_page>;
    static constexpr int log2_struct = NextLogTwo<sizeof(Struct_type)>::value;
    void *data_ptr;

protected:
    using Base::Get_Data_Ptr;using Base::block_bits;

public:
    SPGrid_Colored_Allocator(Base& base_allocator):Base(std::array<ucoord_t,dim>{}){data_ptr=base_allocator.Get_Data_Ptr();};
    template<class Field_type=Struct_type> using Array_mask=SPGrid_Mask<log2_struct+dim,NextLogTwo<sizeof(Field_type)>::value,dim,log2_page>;
    template<class Field_type=Struct_type> using Array_type=SPGrid_Array<Field_type,Array_mask<Field_type>>;

    template<int color,class T1,class Field_type> typename std::enable_if<std::is_same<T1,Struct_type>::value,Array_type<Field_type>>::type
    Get_Array(Field_type T1::* field) const
    {
        constexpr size_t color_offset = color<<(block_bits-dim+NextLogTwo<sizeof(Field_type)>::value);
        size_t offset=(OffsetOfMember(field)<<block_bits)+color_offset;
        void* offset_ptr=reinterpret_cast<void*>(reinterpret_cast<uint64_t>(data_ptr)+offset);
        return Array_type<Field_type>(offset_ptr,*this);
    }

    template<int color,class T1,class Field_type> typename std::enable_if<std::is_same<T1,Struct_type>::value,Array_type<const Field_type>>::type
    Get_Const_Array(Field_type T1::* field) const
    {
        constexpr size_t color_offset = color<<(block_bits-dim+NextLogTwo<sizeof(Field_type)>::value);
        size_t offset=(OffsetOfMember(field)<<block_bits)+color_offset;
        void* offset_ptr=reinterpret_cast<void*>(reinterpret_cast<uint64_t>(data_ptr)+offset);
        return Array_type<const Field_type>(offset_ptr,*this);
    }

    ~SPGrid_Colored_Allocator(){};
};
}
#endif
