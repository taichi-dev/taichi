//#####################################################################
// Copyright (c) 2012, Eftychios Sifakis, Sean Bauer
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Set_h__
#define __SPGrid_Set_h__

#include <SPGrid/Core/SPGrid_Page_Map.h>

namespace SPGrid{
//#####################################################################
// Class SPGrid_Set
//#####################################################################
template<class T_ARRAY>
class SPGrid_Set
{
    enum {d=T_ARRAY::dim};
    typedef typename T_ARRAY::DATA T;

    SPGrid_Page_Map& pagemap;
    T_ARRAY array;

public:
    SPGrid_Set(SPGrid_Page_Map& pagemap_input,T_ARRAY array_input)
        :pagemap(pagemap_input),array(array_input)
    {}    

    // Note: the following are thread-safe only at coordinate granularity.
    // Different threads can safely process entries of the same page, but cannot safely mask different bits of the same entry
    void Mask(const std::array<ucoord_t,d>& coord,const T mask)
    {
        static_assert(std::is_integral<T>::value,"Masking allowed only with integral types in SPGrid_Set");
        auto linear_offset=T_ARRAY::MASK::Linear_Offset(coord);
        pagemap.Set_Page(linear_offset);
        array(linear_offset) |= mask;
    }

    void Mask(const uint64_t linear_offset,const T mask)
    {
        static_assert(std::is_integral<T>::value,"Masking allowed only with integral types in SPGrid_Set");
        pagemap.Set_Page(linear_offset);
        array(linear_offset) |= mask;
    }

//#####################################################################
};
}
#endif
