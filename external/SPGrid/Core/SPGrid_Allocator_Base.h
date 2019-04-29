//#####################################################################
// Copyright (c) 2012-2013, Eftychios Sifakis, Sean Bauer
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Allocator_Base_h__
#define __SPGrid_Allocator_Base_h__

#include <type_traits>
#include <SPGrid/Core/SPGrid_Mask.h>
#include <SPGrid/Core/SPGrid_Array.h>

namespace SPGrid{
//#####################################################################
// Class SPGrid_Allocator_Base
//#####################################################################

template<int log2_struct,int dim,int log2_page=12>
class SPGrid_Allocator_Base: public SPGrid_Geometry<dim>, public SPGrid_Mask_base<log2_struct,dim,log2_page>
{
    using T_Geometry_Base = SPGrid_Geometry<dim>;
    using T_Mask_Base = SPGrid_Mask_base<log2_struct,dim,log2_page>;

    using T_Geometry_Base::Padded_Volume;using T_Mask_Base::block_xbits;using T_Mask_Base::block_ybits;
    //using T_Mask_Base::elements_per_block;

protected:
    using T_Mask_Base::block_bits;

public:
    // Make the allocator class noncopyable
    SPGrid_Allocator_Base(const SPGrid_Allocator_Base&) = delete;
    SPGrid_Allocator_Base& operator=(const SPGrid_Allocator_Base&) = delete;

    SPGrid_Allocator_Base(const ucoord_t xsize_input, const ucoord_t ysize_input, const ucoord_t zsize_input)
        :SPGrid_Geometry<dim>(xsize_input,ysize_input,zsize_input,block_xbits,block_ybits,T_Mask_Base::block_zbits)
    {
        static_assert(dim==3,"Dimension mismatch");
        Check_Compliance();
        data_ptr=Raw_Allocate(Padded_Volume()<<log2_struct);
    }

    SPGrid_Allocator_Base(const ucoord_t xsize_input, const ucoord_t ysize_input)
        :SPGrid_Geometry<dim>(xsize_input,ysize_input,block_xbits,block_ybits)
    {
        static_assert(dim==2,"Dimension mismatch");
        Check_Compliance();
        data_ptr=Raw_Allocate(Padded_Volume()<<log2_struct);
    }
    
    SPGrid_Allocator_Base(const std::array<ucoord_t,3> size_in)
        :SPGrid_Geometry<dim>(size_in[0],size_in[1],size_in[2],block_xbits,block_ybits,T_Mask_Base::block_zbits)
    {
        static_assert(dim==3,"Dimension mismatch");
        Check_Compliance();
        data_ptr=Raw_Allocate(Padded_Volume()<<log2_struct);
    }

    SPGrid_Allocator_Base(const std::array<ucoord_t,2> size_in)
        :SPGrid_Geometry<dim>(size_in[0],size_in[1],block_xbits,block_ybits)
    {
        static_assert(dim==2,"Dimension mismatch");
        Check_Compliance();
        data_ptr=Raw_Allocate(Padded_Volume()<<log2_struct);
    }

    ~SPGrid_Allocator_Base()
    {Raw_Deallocate(data_ptr,Padded_Volume()<<log2_struct);}

    void Validate(uint64_t *page_mask_array) const
    {Validate_Memory_Use((Padded_Volume()<<log2_struct)>>12,Get_Data_Ptr(),page_mask_array);}

protected:    
    inline void* Get_Data_Ptr() const {return data_ptr;}

private:
    void *data_ptr;

//#####################################################################
};
}
#endif
