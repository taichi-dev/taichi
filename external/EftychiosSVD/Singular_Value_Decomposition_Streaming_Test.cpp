//#####################################################################
// Copyright (c) 2010-2011, Eftychios Sifakis.
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

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>

// #define PERFORM_CORRECTNESS_TEST

#ifdef PERFORM_CORRECTNESS_TEST
#include <vector>
#endif

#include "PTHREAD_QUEUE.h"
#include "Singular_Value_Decomposition_Helper.h"
using namespace Singular_Value_Decomposition;
using namespace PhysBAM;
extern PhysBAM::PTHREAD_QUEUE* pthread_queue;

struct timeval starttime,stoptime;
void start_timer(){gettimeofday(&starttime,NULL);}
void stop_timer(){gettimeofday(&stoptime,NULL);}
double get_time(){return (double)stoptime.tv_sec-(double)starttime.tv_sec+(double)1e-6*(double)stoptime.tv_usec-(double)1e-6*(double)starttime.tv_usec;}

int main(int argc,char *argv[]){
    
    typedef float T;
    // static const int size=1024*1024;
    static const int size=64*1024*1024;
    // static const int size=65536;

    T *a11,*a21,*a31,*a12,*a22,*a32,*a13,*a23,*a33;
    T *u11,*u21,*u31,*u12,*u22,*u32,*u13,*u23,*u33;
    T *v11,*v21,*v31,*v12,*v22,*v32,*v13,*v23,*v33;
    T *sigma1,*sigma2,*sigma3;

    if(argc!=2){printf("Must specify number of threads\n");exit(1);}
    int number_of_threads=atoi(argv[1]);
    printf("Using %d threads\n",number_of_threads);
    pthread_queue=new PhysBAM::PTHREAD_QUEUE(number_of_threads);  

    // Allocate data
    printf("Allocation");start_timer();

    Singular_Value_Decomposition_Size_Specific_Helper<T,size>::Allocate_Data(
        a11,a21,a31,a12,a22,a32,a13,a23,a33,
        u11,u21,u31,u12,u22,u32,u13,u23,u33,
        v11,v21,v31,v12,v22,v32,v13,v23,v33,
        sigma1,sigma2,sigma3);

    stop_timer();printf(" [Seconds: %g]\n",get_time());

    // Initialize data
    printf("Initialization");start_timer();

    Singular_Value_Decomposition_Size_Specific_Helper<T,size>::Initialize_Data(
        a11,a21,a31,a12,a22,a32,a13,a23,a33,
        u11,u21,u31,u12,u22,u32,u13,u23,u33,
        v11,v21,v31,v12,v22,v32,v13,v23,v33,
        sigma1,sigma2,sigma3);

    Singular_Value_Decomposition_Size_Specific_Helper<T,size> test(
        a11,a21,a31,a12,a22,a32,a13,a23,a33,
        u11,u21,u31,u12,u22,u32,u13,u23,u33,
        v11,v21,v31,v12,v22,v32,v13,v23,v33,
        sigma1,sigma2,sigma3);

    stop_timer();printf(" [Seconds: %g]\n",get_time());

#ifdef PERFORM_CORRECTNESS_TEST
    printf("Running correctness benchmark");
    start_timer();
    test.Run_Parallel(number_of_threads); 
    stop_timer();
    printf(" [Seconds: %g]\n",get_time());

    T A[3][3],U[3][3],V[3][3],Sigma[3],A_rotated[3][3],A_reconstructed[3][3];
    
    std::vector<T> max_reconstruction_error(size);
    std::vector<T> max_off_diagonal_element(size);

    for(int i=0;i<size;i++){

      A[0][0]=a11[i];A[0][1]=a12[i];A[0][2]=a13[i];A[1][0]=a21[i];A[1][1]=a22[i];A[1][2]=a23[i];A[2][0]=a31[i];A[2][1]=a32[i];A[2][2]=a33[i];
      U[0][0]=u11[i];U[0][1]=u12[i];U[0][2]=u13[i];U[1][0]=u21[i];U[1][1]=u22[i];U[1][2]=u23[i];U[2][0]=u31[i];U[2][1]=u32[i];U[2][2]=u33[i];
      V[0][0]=v11[i];V[0][1]=v12[i];V[0][2]=v13[i];V[1][0]=v21[i];V[1][1]=v22[i];V[1][2]=v23[i];V[2][0]=v31[i];V[2][1]=v32[i];V[2][2]=v33[i];
      Sigma[0]=sigma1[i];Sigma[1]=sigma2[i];Sigma[2]=sigma3[i];

      for(int k=0;k<3;k++)
        for(int l=0;l<3;l++){
          A_rotated[k][l]=0.;
          for(int m=0;m<3;m++)
            for(int n=0;n<3;n++)
              A_rotated[k][l]+=U[m][k]*A[m][n]*V[n][l];}

      max_off_diagonal_element[i]=0.;
      if(A_rotated[0][1]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=A_rotated[0][1]; else if(-A_rotated[0][1]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=-A_rotated[0][1];
      if(A_rotated[0][2]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=A_rotated[0][2]; else if(-A_rotated[0][2]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=-A_rotated[0][2];
      if(A_rotated[1][0]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=A_rotated[1][0]; else if(-A_rotated[1][0]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=-A_rotated[1][0];
      if(A_rotated[1][2]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=A_rotated[1][2]; else if(-A_rotated[1][2]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=-A_rotated[1][2];
      if(A_rotated[2][0]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=A_rotated[2][0]; else if(-A_rotated[2][0]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=-A_rotated[2][0];
      if(A_rotated[2][1]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=A_rotated[2][1]; else if(-A_rotated[2][1]>max_off_diagonal_element[i]) max_off_diagonal_element[i]=-A_rotated[2][1];

      max_reconstruction_error[i]=0.;
      for(int k=0;k<3;k++)
        for(int l=0;l<3;l++){
          A_reconstructed[k][l]=0.;
          for(int m=0;m<3;m++)
            A_reconstructed[k][l]+=U[k][m]*Sigma[m]*V[l][m];
          T reconstruction_error=A[k][l]-A_reconstructed[k][l];
          if(reconstruction_error>max_reconstruction_error[i]) max_reconstruction_error[i]=reconstruction_error; else if(-reconstruction_error>max_reconstruction_error[i]) max_reconstruction_error[i]=-reconstruction_error;}

    }

    sort(max_off_diagonal_element.begin(),max_off_diagonal_element.end());
    std::cout<<"Maximum off-diagonal element, after approximate diagonalization"<<std::endl;
    std::cout<<"  Worst case (largest value)    : "<<max_off_diagonal_element[size-1]<<std::endl;
    std::cout<<"  99.9-percentile largest value : "<<max_off_diagonal_element[999*((size-1)/1000)]<<std::endl;
    std::cout<<"  99-percentile largest value   : "<<max_off_diagonal_element[99*((size-1)/100)]<<std::endl;

    sort(max_reconstruction_error.begin(),max_reconstruction_error.end());
    std::cout<<"Maximum reconstruction error, i.e. maximum absolute entry of A-U*Sigma*Transpose(V)"<<std::endl;
    std::cout<<"  Worst case (largest value)    : "<<max_reconstruction_error[size-1]<<std::endl;
    std::cout<<"  99.9-percentile largest value : "<<max_reconstruction_error[999*((size-1)/1000)]<<std::endl;
    std::cout<<"  99-percentile largest value   : "<<max_reconstruction_error[99*((size-1)/100)]<<std::endl;

#else
    // Run performance benchmark
    while(1){
        printf("Running performance benchmark");

        start_timer();
 	test.Run_Parallel(number_of_threads); 
        stop_timer();

        printf(" [Seconds: %g]\n",get_time());
     }
#endif

    return 0;
}
