//#####################################################################
// Copyright 2012-2013, Sean Bauer, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
// Class SPGrid_Array
//#####################################################################
#ifndef __SPGrid_Array_h__
#define __SPGrid_Array_h__

//#define SPGRID_CHECK_BOUNDS

#include <SPGrid/Core/SPGrid_Geometry.h>

namespace SPGrid{

template<class T,class T_MASK>
class SPGrid_Array
{
public:
    enum {dim=T_MASK::dim};
    typedef T DATA;
    typedef T_MASK MASK;

private:
    void* const data_ptr;

public:
    const SPGrid_Geometry<dim>& geometry;

    SPGrid_Array(void* const data_ptr_input, const SPGrid_Geometry<dim>& geometry_input)
        :data_ptr(data_ptr_input),geometry(geometry_input)
    {}

    inline T& operator()(const std::array<ucoord_t,3>& coord)
    {
        static_assert(dim==3,"Dimension mismatch");
#ifdef SPGRID_CHECK_BOUNDS
        geometry.Check_Bounds(coord[0],coord[1],coord[2]);
#endif        
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Linear_Offset(coord));
    }
    
    inline T& operator()(const std::array<ucoord_t,2>& coord)
    {
        static_assert(dim==2,"Dimension mismatch");
#ifdef SPGRID_CHECK_BOUNDS
        geometry.Check_Bounds(coord[0],coord[1]);
#endif        
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Linear_Offset(coord));
    }

    inline T& operator()(const std::array<scoord_t,3>& coord)
    {
        static_assert(dim==3,"Dimension mismatch");
#ifdef SPGRID_CHECK_BOUNDS
        geometry.Check_Bounds(coord[0],coord[1],coord[2]);
#endif        
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Linear_Offset(coord));
    }
    
    inline T& operator()(const std::array<scoord_t,2>& coord)
    {
        static_assert(dim==2,"Dimension mismatch");
#ifdef SPGRID_CHECK_BOUNDS
        geometry.Check_Bounds(coord[0],coord[1]);
#endif        
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Linear_Offset(coord));
    }
    
    inline T& operator()(const ucoord_t i,const ucoord_t j,const ucoord_t k)
    {
        static_assert(dim==3,"Dimension mismatch");
#ifdef SPGRID_CHECK_BOUNDS
        geometry.Check_Bounds(i,j,k);
#endif        
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Linear_Offset(i,j,k));
    }
    
    inline T& operator()(const ucoord_t i,const ucoord_t j)
    {
        static_assert(dim==2,"Dimension mismatch");
#ifdef SPGRID_CHECK_BOUNDS
        geometry.Check_Bounds(i,j);
#endif        
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Linear_Offset(i,j));
    }

    inline T& operator()(const uint64_t offset)
    {
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+offset);
    }

    template<int di, int dj, int dk>
    inline T& operator()(const uint64_t offset)
    {
        static_assert(dim==3,"Dimension mismatch");
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Packed_Offset<di,dj,dk>(offset));
    }

    template<int di, int dj>
    inline T& operator()(const uint64_t offset)
    {
        static_assert(dim==2,"Dimension mismatch");
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Packed_Offset<di,dj>(offset));
    }

    // Debug_Get() functions operate like the operator parenthesis, but also check if the memory address is resident
    T& Debug_Get(const ucoord_t i,const ucoord_t j,const ucoord_t k)
    {
        static_assert(dim==3,"Dimension mismatch");
#ifdef SPGRID_CHECK_BOUNDS
        geometry.Check_Bounds(i,j,k);
#endif        
        T* addr=reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Linear_Offset(i,j,k));
        Check_Address_Resident(addr);
        return *addr;
    }

    T& Debug_Get(const ucoord_t i,const ucoord_t j)
    {
        static_assert(dim==2,"Dimension mismatch");
#ifdef SPGRID_CHECK_BOUNDS
        geometry.Check_Bounds(i,j);
#endif        
        T* addr=reinterpret_cast<T*>(reinterpret_cast<uint64_t>(data_ptr)+T_MASK::Linear_Offset(i,j));
        Check_Address_Resident(addr);
        return *addr;
    }

    inline const void* Get_Data_Ptr() const { return data_ptr; }
    inline void* Get_Data_Ptr() { return data_ptr; }

//#####################################################################
};
}

#endif
