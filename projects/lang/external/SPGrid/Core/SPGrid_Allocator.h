//#####################################################################
// Copyright (c) 2012-2013, Eftychios Sifakis, Sean Bauer
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Allocator_h__
#define __SPGrid_Allocator_h__

#include <SPGrid/Core/SPGrid_Allocator_Base.h>

namespace SPGrid{
//#####################################################################
// Class SPGrid_Allocator
//#####################################################################

template<class Struct_type,int dim,int log2_page=12>
class SPGrid_Allocator: public SPGrid_Allocator_Base<NextLogTwo<sizeof(Struct_type)>::value,dim,log2_page>
{
    using Base = SPGrid_Allocator_Base<NextLogTwo<sizeof(Struct_type)>::value,dim,log2_page>;
    static constexpr int log2_struct = NextLogTwo<sizeof(Struct_type)>::value;

protected:
    using Base::block_bits;
public:
    using Base::Get_Data_Ptr;
    using Base::Base; // Inherit constructors from base

    template<class Field_type=Struct_type> using Array_mask=SPGrid_Mask<log2_struct,NextLogTwo<sizeof(Field_type)>::value,dim,log2_page>;
    template<class Field_type=Struct_type> using Array_type=SPGrid_Array<Field_type,Array_mask<Field_type>>;

    template<class T1,class Field_type> typename std::enable_if<std::is_same<T1,Struct_type>::value,Array_type<Field_type>>::type
    Get_Array(Field_type T1::* field)
    {
        size_t offset=OffsetOfMember(field)<<block_bits;
        void* offset_ptr=reinterpret_cast<void*>(reinterpret_cast<uint64_t>(Get_Data_Ptr())+offset);
        return Array_type<Field_type>(offset_ptr,*this);
    }

    template<class T1,class Field_type> typename std::enable_if<std::is_same<T1,Struct_type>::value,Array_type<const Field_type>>::type
    Get_Array(Field_type T1::* field) const
    {
        size_t offset=OffsetOfMember(field)<<block_bits;
        void* offset_ptr=reinterpret_cast<void*>(reinterpret_cast<uint64_t>(Get_Data_Ptr())+offset);
        return Array_type<const Field_type>(offset_ptr,*this);
    }

    template<class T1,class Field_type> typename std::enable_if<std::is_same<T1,Struct_type>::value,Array_type<const Field_type>>::type
    Get_Const_Array(Field_type T1::* field) const
    {
        size_t offset=OffsetOfMember(field)<<block_bits;
        void* offset_ptr=reinterpret_cast<void*>(reinterpret_cast<uint64_t>(Get_Data_Ptr())+offset);
        return Array_type<const Field_type>(offset_ptr,*this);
    }

    Array_type<> Get_Array()
    {return Array_type<>(Get_Data_Ptr(),*this);}

    Array_type<const Struct_type> Get_Array() const
    {return Array_type<const Struct_type>(Get_Data_Ptr(),*this);}

    Array_type<const Struct_type> Get_Const_Array() const
    {return Array_type<const Struct_type>(Get_Data_Ptr(),*this);}

//#####################################################################
};
}
#endif
