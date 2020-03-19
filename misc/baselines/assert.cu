// clang-format off
#include <cstdio>
#include <cassert>

__global__ void init_random_numbers(unsigned int seed) {
  printf("seed = %d\n", seed);
  atomicAdd((int *)(12312433432), 123);
  atomicAdd((float *)(12312433432), 123.0f);
  __threadfence_block(); // membar.cta
  __threadfence(); // membar.gl
  __threadfence_system();  // membar.sys
  assert(seed != 0);
}


// How LLVM deals with CUDA kernels with huge structs as parameters:
struct Arg {
  float x[128];
  int y[128];
};


/* llvm IR of function below:
 ; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z20test_struct_argument3Arg(%struct.Arg* byval align 4) #0 {
  %2 = alloca %printf_args.0
  %3 = getelementptr inbounds %struct.Arg, %struct.Arg* %0, i32 0, i32 0
  %4 = getelementptr inbounds [128 x float], [128 x float]* %3, i64 0, i64 123
  %5 = load float, float* %4, align 4
  %6 = fpext float %5 to double
  %7 = getelementptr inbounds %struct.Arg, %struct.Arg* %0, i32 0, i32 1
  %8 = getelementptr inbounds [128 x i32], [128 x i32]* %7, i64 0, i64 53
  %9 = load i32, i32* %8, align 4
  %10 = getelementptr inbounds %printf_args.0, %printf_args.0* %2, i32 0, i32 0
  store double %6, double* %10, align 8
  %11 = getelementptr inbounds %printf_args.0, %printf_args.0* %2, i32 0, i32 1
  store i32 %9, i32* %11, align 4
  %12 = bitcast %printf_args.0* %2 to i8*
  %13 = call i32 @vprintf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str3, i32 0, i32 0), i8* %12)
  ret void
}
*/
__global__ void test_struct_argument(Arg arg) {
  printf("%f %d\n", arg.x[123], arg.y[53]);
}


/* llvm IR of function below:
; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z24test_struct_argument_ptrP3Arg(%struct.Arg*) #0 {
%2 = alloca %struct.Arg*, align 8
%3 = alloca %printf_args.1
store %struct.Arg* %0, %struct.Arg** %2, align 8
%4 = load %struct.Arg*, %struct.Arg** %2, align 8
%5 = getelementptr inbounds %struct.Arg, %struct.Arg* %4, i32 0, i32 0
%6 = getelementptr inbounds [128 x float], [128 x float]* %5, i64 0, i64 123
%7 = load float, float* %6, align 4
%8 = fpext float %7 to double
%9 = load %struct.Arg*, %struct.Arg** %2, align 8
%10 = getelementptr inbounds %struct.Arg, %struct.Arg* %9, i32 0, i32 1
%11 = getelementptr inbounds [128 x i32], [128 x i32]* %10, i64 0, i64 53
%12 = load i32, i32* %11, align 4
%13 = getelementptr inbounds %printf_args.1, %printf_args.1* %3, i32 0, i32 0
store double %8, double* %13, align 8
%14 = getelementptr inbounds %printf_args.1, %printf_args.1* %3, i32 0, i32 1
store i32 %12, i32* %14, align 4
%15 = bitcast %printf_args.1* %3 to i8*
%16 = call i32 @vprintf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str3, i32 0, i32 0), i8* %15)
ret void
}
*/
__global__ void test_struct_argument_ptr(Arg *arg) {
  printf("%f %d\n", arg->x[123], arg->y[53]);
}

int main() {
  init_random_numbers<<<1024, 1024>>>(1);
  Arg arg;
  test_struct_argument<<<1, 1>>>(arg);
  return 0;
}
// clang-format on
