//#####################################################################
// Copyright (c) 2009-2011, Eftychios Sifakis.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//   * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
//     other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//#####################################################################

#ifndef __Singular_Value_Decomposition_Helper__
#define __Singular_Value_Decomposition_Helper__

namespace Singular_Value_Decomposition{

template<class T,int size>
class Singular_Value_Decomposition_Size_Specific_Helper
{
    T* const a11,* const a21,* const a31,* const a12,* const a22,* const a32,* const a13,* const a23,* const a33;
    T* const u11,* const u21,* const u31,* const u12,* const u22,* const u32,* const u13,* const u23,* const u33;
    T* const v11,* const v21,* const v31,* const v12,* const v22,* const v32,* const v13,* const v23,* const v33;
    T* const sigma1,* const sigma2,* const sigma3;

public:
    explicit Singular_Value_Decomposition_Size_Specific_Helper(
        T* const a11_input,T* const a21_input,T* const a31_input,
        T* const a12_input,T* const a22_input,T* const a32_input,
        T* const a13_input,T* const a23_input,T* const a33_input,
        T* const u11_input,T* const u21_input,T* const u31_input,
        T* const u12_input,T* const u22_input,T* const u32_input,
        T* const u13_input,T* const u23_input,T* const u33_input,
        T* const v11_input,T* const v21_input,T* const v31_input,
        T* const v12_input,T* const v22_input,T* const v32_input,
        T* const v13_input,T* const v23_input,T* const v33_input,
        T* const sigma1_input,T* const sigma2_input,T* const sigma3_input)
        :a11(a11_input),a21(a21_input),a31(a31_input),a12(a12_input),a22(a22_input),a32(a32_input),a13(a13_input),a23(a23_input),a33(a33_input),
        u11(u11_input),u21(u21_input),u31(u31_input),u12(u12_input),u22(u22_input),u32(u32_input),u13(u13_input),u23(u23_input),u33(u33_input),
        v11(v11_input),v21(v21_input),v31(v31_input),v12(v12_input),v22(v22_input),v32(v32_input),v13(v13_input),v23(v23_input),v33(v33_input),
        sigma1(sigma1_input),sigma2(sigma2_input),sigma3(sigma3_input)
    {}

    void Run()
    {Run_Index_Range(0,size);}
  
//#####################################################################
    static void Allocate_Data(
        T*& a11,T*& a21,T*& a31,T*& a12,T*& a22,T*& a32,T*& a13,T*& a23,T*& a33,
        T*& u11,T*& u21,T*& u31,T*& u12,T*& u22,T*& u32,T*& u13,T*& u23,T*& u33,
        T*& v11,T*& v21,T*& v31,T*& v12,T*& v22,T*& v32,T*& v13,T*& v23,T*& v33,
        T*& sigma1,T*& sigma2,T*& sigma3);
    static void Initialize_Data(
        T*& a11,T*& a21,T*& a31,T*& a12,T*& a22,T*& a32,T*& a13,T*& a23,T*& a33,
        T*& u11,T*& u21,T*& u31,T*& u12,T*& u22,T*& u32,T*& u13,T*& u23,T*& u33,
        T*& v11,T*& v21,T*& v31,T*& v12,T*& v22,T*& v32,T*& v13,T*& v23,T*& v33,
        T*& sigma1,T*& sigma2,T*& sigma3);
    void Run_Parallel(const int number_of_partitions);
    void Run_Index_Range(const int imin, const int imax_plus_one);
//#####################################################################
};
}
#endif
