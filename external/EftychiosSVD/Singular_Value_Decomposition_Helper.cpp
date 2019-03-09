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

#include <cmath>
#include <algorithm>
#include <iostream>
#include "PTHREAD_QUEUE.h"

#define COMPUTE_V_AS_MATRIX
// #define COMPUTE_V_AS_QUATERNION
#define COMPUTE_U_AS_MATRIX
// #define COMPUTE_U_AS_QUATERNION

#include "Singular_Value_Decomposition_Preamble.hpp"
#include "Singular_Value_Decomposition_Helper.h"

// #define USE_SCALAR_IMPLEMENTATION
// #define USE_SSE_IMPLEMENTATION
// #define USE_AVX_IMPLEMENTATION
// #define USE_AVX512_IMPLEMENTATION
// #define PRINT_DEBUGGING_OUTPUT


using namespace Singular_Value_Decomposition;
using namespace PhysBAM;
extern PTHREAD_QUEUE* pthread_queue;

//#####################################################################
// Function Allocate_Data
//#####################################################################
template<class T,int size> void Singular_Value_Decomposition_Size_Specific_Helper<T,size>::
Allocate_Data(T*& a11,T*& a21,T*& a31,T*& a12,T*& a22,T*& a32,T*& a13,T*& a23,T*& a33,
    T*& u11,T*& u21,T*& u31,T*& u12,T*& u22,T*& u32,T*& u13,T*& u23,T*& u33,
    T*& v11,T*& v21,T*& v31,T*& v12,T*& v22,T*& v32,T*& v13,T*& v23,T*& v33,
    T*& sigma1,T*& sigma2,T*& sigma3)
{
    a11=new T[size];
    a21=new T[size];
    a31=new T[size];
    a12=new T[size];
    a22=new T[size];
    a32=new T[size];
    a13=new T[size];
    a23=new T[size];
    a33=new T[size];

    u11=new T[size];
    u21=new T[size];
    u31=new T[size];
    u12=new T[size];
    u22=new T[size];
    u32=new T[size];
    u13=new T[size];
    u23=new T[size];
    u33=new T[size];

    v11=new T[size];
    v21=new T[size];
    v31=new T[size];
    v12=new T[size];
    v22=new T[size];
    v32=new T[size];
    v13=new T[size];
    v23=new T[size];
    v33=new T[size];

    sigma1=new T[size];
    sigma2=new T[size];
    sigma3=new T[size];
}
//#####################################################################
// Function Initialize_Data
//#####################################################################
template<class T,int size> void Singular_Value_Decomposition_Size_Specific_Helper<T,size>::
Initialize_Data(T*& a11,T*& a21,T*& a31,T*& a12,T*& a22,T*& a32,T*& a13,T*& a23,T*& a33,
    T*& u11,T*& u21,T*& u31,T*& u12,T*& u22,T*& u32,T*& u13,T*& u23,T*& u33,
    T*& v11,T*& v21,T*& v31,T*& v12,T*& v22,T*& v32,T*& v13,T*& v23,T*& v33,
    T*& sigma1,T*& sigma2,T*& sigma3)
{
    srand(1);
    for(int i=0;i<size;i++){
        a11[i]=2.*(T)rand()/(T)RAND_MAX-1.;a21[i]=2.*(T)rand()/(T)RAND_MAX-1.;a31[i]=2.*(T)rand()/(T)RAND_MAX-1.;
        a12[i]=2.*(T)rand()/(T)RAND_MAX-1.;a22[i]=2.*(T)rand()/(T)RAND_MAX-1.;a32[i]=2.*(T)rand()/(T)RAND_MAX-1.;
        a13[i]=2.*(T)rand()/(T)RAND_MAX-1.;a23[i]=2.*(T)rand()/(T)RAND_MAX-1.;a33[i]=2.*(T)rand()/(T)RAND_MAX-1.;

        T one_over_frobenius_norm=(T)1./sqrt(
            (double)a11[i]*(double)a11[i]+(double)a12[i]*(double)a12[i]+(double)a13[i]*(double)a13[i]+
            (double)a21[i]*(double)a21[i]+(double)a22[i]*(double)a22[i]+(double)a23[i]*(double)a23[i]+
            (double)a31[i]*(double)a31[i]+(double)a32[i]*(double)a32[i]+(double)a33[i]*(double)a33[i]);

        a11[i]*=one_over_frobenius_norm;
        a12[i]*=one_over_frobenius_norm;
        a13[i]*=one_over_frobenius_norm;
        a21[i]*=one_over_frobenius_norm;
        a22[i]*=one_over_frobenius_norm;
        a23[i]*=one_over_frobenius_norm;
        a31[i]*=one_over_frobenius_norm;
        a32[i]*=one_over_frobenius_norm;
        a33[i]*=one_over_frobenius_norm;
    }
}
//#####################################################################
// Function Run_Parallel
//#####################################################################
namespace{
template<class T,int size>
struct Singular_Value_Decomposition_Size_Specific_Thread_Helper:public PhysBAM::PTHREAD_QUEUE::TASK
{
    Singular_Value_Decomposition_Size_Specific_Helper<T,size>* const obj;
    const int imin,imax_plus_one;
    Singular_Value_Decomposition_Size_Specific_Thread_Helper(
        Singular_Value_Decomposition_Size_Specific_Helper<T,size>* const obj_input,const int imin_input,const int imax_plus_one_input)
        :obj(obj_input),imin(imin_input),imax_plus_one(imax_plus_one_input) {}
    void Run(){obj->Run_Index_Range(imin,imax_plus_one);}
};
}
template<class T,int size> void Singular_Value_Decomposition_Size_Specific_Helper<T,size>::
Run_Parallel(const int number_of_partitions)
{
    for(int partition=0;partition<number_of_partitions;partition++){
        int imin=(size/number_of_partitions)*partition+std::min(size%number_of_partitions,partition);
        int imax_plus_one=(size/number_of_partitions)*(partition+1)+std::min(size%number_of_partitions,partition+1);
 	Singular_Value_Decomposition_Size_Specific_Thread_Helper<T,size>* task=new Singular_Value_Decomposition_Size_Specific_Thread_Helper<T,size>(this,imin,imax_plus_one);
 	pthread_queue->Queue(task);}
    pthread_queue->Wait();    
}
//#####################################################################
// Function Run_Index_Range
//#####################################################################
template<class T,int size> void Singular_Value_Decomposition_Size_Specific_Helper<T,size>::
Run_Index_Range(const int imin,const int imax_plus_one)
{   

#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

#ifdef USE_SSE_IMPLEMENTATION
#define STEP_SIZE 4
#endif
#ifdef USE_AVX_IMPLEMENTATION
#define STEP_SIZE 8
#endif
#ifdef USE_AVX512_IMPLEMENTATION
#define STEP_SIZE 16
#endif
#ifdef USE_SCALAR_IMPLEMENTATION
#define STEP_SIZE 1
#endif

    for(int index=imin;index<imax_plus_one;index+=STEP_SIZE){
        
    ENABLE_SCALAR_IMPLEMENTATION(Sa11.f=a11[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va11=_mm_loadu_ps(a11+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va11=_mm256_loadu_ps(a11+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va11=_mm512_loadu_ps(a11+index);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa21.f=a21[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va21=_mm_loadu_ps(a21+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va21=_mm256_loadu_ps(a21+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va21=_mm512_loadu_ps(a21+index);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa31.f=a31[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va31=_mm_loadu_ps(a31+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va31=_mm256_loadu_ps(a31+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va31=_mm512_loadu_ps(a31+index);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa12.f=a12[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va12=_mm_loadu_ps(a12+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va12=_mm256_loadu_ps(a12+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va12=_mm512_loadu_ps(a12+index);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa22.f=a22[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va22=_mm_loadu_ps(a22+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va22=_mm256_loadu_ps(a22+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va22=_mm512_loadu_ps(a22+index);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa32.f=a32[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va32=_mm_loadu_ps(a32+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va32=_mm256_loadu_ps(a32+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va32=_mm512_loadu_ps(a32+index);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa13.f=a13[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va13=_mm_loadu_ps(a13+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va13=_mm256_loadu_ps(a13+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va13=_mm512_loadu_ps(a13+index);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa23.f=a23[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va23=_mm_loadu_ps(a23+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va23=_mm256_loadu_ps(a23+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va23=_mm512_loadu_ps(a23+index);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa33.f=a33[index];)                                      ENABLE_SSE_IMPLEMENTATION(Va33=_mm_loadu_ps(a33+index);)                                  ENABLE_AVX_IMPLEMENTATION(Va33=_mm256_loadu_ps(a33+index);)                               ENABLE_AVX512_IMPLEMENTATION(Va33=_mm512_loadu_ps(a33+index);)
        
#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"
        
    ENABLE_SCALAR_IMPLEMENTATION(u11[index]=Su11.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u11+index,Vu11);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u11+index,Vu11);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u11+index,Vu11);)
    ENABLE_SCALAR_IMPLEMENTATION(u21[index]=Su21.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u21+index,Vu21);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u21+index,Vu21);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u21+index,Vu21);)
    ENABLE_SCALAR_IMPLEMENTATION(u31[index]=Su31.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u31+index,Vu31);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u31+index,Vu31);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u31+index,Vu31);)
    ENABLE_SCALAR_IMPLEMENTATION(u12[index]=Su12.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u12+index,Vu12);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u12+index,Vu12);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u12+index,Vu12);)
    ENABLE_SCALAR_IMPLEMENTATION(u22[index]=Su22.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u22+index,Vu22);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u22+index,Vu22);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u22+index,Vu22);)
    ENABLE_SCALAR_IMPLEMENTATION(u32[index]=Su32.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u32+index,Vu32);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u32+index,Vu32);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u32+index,Vu32);)
    ENABLE_SCALAR_IMPLEMENTATION(u13[index]=Su13.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u13+index,Vu13);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u13+index,Vu13);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u13+index,Vu13);)
    ENABLE_SCALAR_IMPLEMENTATION(u23[index]=Su23.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u23+index,Vu23);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u23+index,Vu23);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u23+index,Vu23);)
    ENABLE_SCALAR_IMPLEMENTATION(u33[index]=Su33.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(u33+index,Vu33);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(u33+index,Vu33);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(u33+index,Vu33);)
        
    ENABLE_SCALAR_IMPLEMENTATION(v11[index]=Sv11.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v11+index,Vv11);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v11+index,Vv11);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v11+index,Vv11);)
    ENABLE_SCALAR_IMPLEMENTATION(v21[index]=Sv21.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v21+index,Vv21);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v21+index,Vv21);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v21+index,Vv21);)
    ENABLE_SCALAR_IMPLEMENTATION(v31[index]=Sv31.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v31+index,Vv31);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v31+index,Vv31);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v31+index,Vv31);)
    ENABLE_SCALAR_IMPLEMENTATION(v12[index]=Sv12.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v12+index,Vv12);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v12+index,Vv12);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v12+index,Vv12);)
    ENABLE_SCALAR_IMPLEMENTATION(v22[index]=Sv22.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v22+index,Vv22);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v22+index,Vv22);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v22+index,Vv22);)
    ENABLE_SCALAR_IMPLEMENTATION(v32[index]=Sv32.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v32+index,Vv32);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v32+index,Vv32);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v32+index,Vv32);)
    ENABLE_SCALAR_IMPLEMENTATION(v13[index]=Sv13.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v13+index,Vv13);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v13+index,Vv13);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v13+index,Vv13);)
    ENABLE_SCALAR_IMPLEMENTATION(v23[index]=Sv23.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v23+index,Vv23);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v23+index,Vv23);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v23+index,Vv23);)
    ENABLE_SCALAR_IMPLEMENTATION(v33[index]=Sv33.f;)                                      ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(v33+index,Vv33);)                                 ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(v33+index,Vv33);)                              ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(v33+index,Vv33);)
        
    ENABLE_SCALAR_IMPLEMENTATION(sigma1[index]=Sa11.f;)                                   ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(sigma1+index,Va11);)                              ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(sigma1+index,Va11);)                           ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(sigma1+index,Va11);)
    ENABLE_SCALAR_IMPLEMENTATION(sigma2[index]=Sa22.f;)                                   ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(sigma2+index,Va22);)                              ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(sigma2+index,Va22);)                           ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(sigma2+index,Va22);)
    ENABLE_SCALAR_IMPLEMENTATION(sigma3[index]=Sa33.f;)                                   ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(sigma3+index,Va33);)                              ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(sigma3+index,Va33);)                           ENABLE_AVX512_IMPLEMENTATION(_mm512_storeu_ps(sigma3+index,Va33);)

    }

#undef STEP_SIZE

}
//#####################################################################
template class Singular_Value_Decomposition_Size_Specific_Helper<float,65536>;
template class Singular_Value_Decomposition_Size_Specific_Helper<float,1048576>;
template class Singular_Value_Decomposition_Size_Specific_Helper<float,4*1048576>;
template class Singular_Value_Decomposition_Size_Specific_Helper<float,16*1048576>;
template class Singular_Value_Decomposition_Size_Specific_Helper<float,64*1048576>;

