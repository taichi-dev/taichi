; ModuleID = 'runtime_bitcode'
source_filename = "runtime.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.RuntimeContext = type { %struct.LLVMRuntime*, [64 x i64], [16 x [8 x i32]], i32, [64 x i64], [64 x i8], i64* }
%struct.LLVMRuntime = type { i8, i64, i8*, i8*, i8* (i8*, i64, i64)*, void (i8*)*, void (i8*, ...)*, i32 (i8*, i64, i8*, %struct.__va_list_tag*)*, i8*, [512 x i8*], [512 x i64], i8*, void (i8*, i32, i32, i8*, void (i8*, i32, i32)*)*, [1024 x %struct.ListManager*], [1024 x %struct.NodeManager*], [1024 x i8*], i8*, %struct.RandState*, %struct.MemRequestQueue*, i8*, void (i8*, i8*)*, void (i8*)*, [2048 x i8], [32 x i64], i32, i64, i8*, i32, i32, i64, i8* }
%struct.__va_list_tag = type { i32, i32, i8*, i8* }
%struct.ListManager = type { [131072 x i8*], i64, i64, i32, i32, i32, %struct.LLVMRuntime* }
%struct.NodeManager = type <{ %struct.LLVMRuntime*, i32, i32, i32, i32, %struct.ListManager*, %struct.ListManager*, %struct.ListManager*, i32, [4 x i8] }>
%struct.RandState = type { i32, i32, i32, i32, i32 }
%struct.MemRequestQueue = type { [65536 x %struct.MemRequest], i32, i32 }
%struct.MemRequest = type { i64, i64, i8*, i64 }
%struct.range_task_helper_context = type { %struct.RuntimeContext*, void (%struct.RuntimeContext*, i8*)*, void (%struct.RuntimeContext*, i8*, i32)*, void (%struct.RuntimeContext*, i8*)*, i64, i32, i32, i32, i32 }

@.str.2 = private unnamed_addr constant [21 x i8] c"step must not be %d\0A\00", align 1

; Function Attrs: alwaysinline nounwind uwtable
define internal i64 @RuntimeContext_get_args(%struct.RuntimeContext* %0, i32 %1) #0 {
  %3 = alloca %struct.RuntimeContext*, align 8
  %4 = alloca i32, align 4
  store %struct.RuntimeContext* %0, %struct.RuntimeContext** %3, align 8
  store i32 %1, i32* %4, align 4
  %5 = load %struct.RuntimeContext*, %struct.RuntimeContext** %3, align 8
  %6 = getelementptr inbounds %struct.RuntimeContext, %struct.RuntimeContext* %5, i32 0, i32 1
  %7 = load i32, i32* %4, align 4
  %8 = sext i32 %7 to i64
  %9 = getelementptr inbounds [64 x i64], [64 x i64]* %6, i64 0, i64 %8
  %10 = load i64, i64* %9, align 8
  ret i64 %10
}

; Function Attrs: alwaysinline nounwind uwtable
define internal %struct.LLVMRuntime* @RuntimeContext_get_runtime(%struct.RuntimeContext* %0) #0 {
  %2 = alloca %struct.RuntimeContext*, align 8
  store %struct.RuntimeContext* %0, %struct.RuntimeContext** %2, align 8
  %3 = load %struct.RuntimeContext*, %struct.RuntimeContext** %2, align 8
  %4 = getelementptr inbounds %struct.RuntimeContext, %struct.RuntimeContext* %3, i32 0, i32 0
  %5 = load %struct.LLVMRuntime*, %struct.LLVMRuntime** %4, align 8
  ret %struct.LLVMRuntime* %5
}

; Function Attrs: alwaysinline nounwind uwtable
define internal i32 @RuntimeContext_get_extra_args(%struct.RuntimeContext* %0, i32 %1, i32 %2) #0 {
  %4 = alloca %struct.RuntimeContext*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store %struct.RuntimeContext* %0, %struct.RuntimeContext** %4, align 8
  store i32 %1, i32* %5, align 4
  store i32 %2, i32* %6, align 4
  %7 = load %struct.RuntimeContext*, %struct.RuntimeContext** %4, align 8
  %8 = getelementptr inbounds %struct.RuntimeContext, %struct.RuntimeContext* %7, i32 0, i32 2
  %9 = load i32, i32* %5, align 4
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds [16 x [8 x i32]], [16 x [8 x i32]]* %8, i64 0, i64 %10
  %12 = load i32, i32* %6, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds [8 x i32], [8 x i32]* %11, i64 0, i64 %13
  %15 = load i32, i32* %14, align 4
  ret i32 %15
}

; Function Attrs: alwaysinline argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: alwaysinline nounwind uwtable
define internal i8* @get_temporary_pointer(%struct.LLVMRuntime* %0, i64 %1) #0 {
  %3 = alloca %struct.LLVMRuntime*, align 8
  %4 = alloca i64, align 8
  store %struct.LLVMRuntime* %0, %struct.LLVMRuntime** %3, align 8
  store i64 %1, i64* %4, align 8
  %5 = load %struct.LLVMRuntime*, %struct.LLVMRuntime** %3, align 8
  %6 = getelementptr inbounds %struct.LLVMRuntime, %struct.LLVMRuntime* %5, i32 0, i32 16
  %7 = load i8*, i8** %6, align 8
  %8 = load i64, i64* %4, align 8
  %9 = getelementptr inbounds i8, i8* %7, i64 %8
  ret i8* %9
}

; Function Attrs: alwaysinline nounwind uwtable
define internal dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4) %0, i32* dereferenceable(4) %1) #0 {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca i32*, align 8
  store i32* %0, i32** %4, align 8
  store i32* %1, i32** %5, align 8
  %6 = load i32*, i32** %5, align 8
  %7 = load i32, i32* %6, align 4
  %8 = load i32*, i32** %4, align 8
  %9 = load i32, i32* %8, align 4
  %10 = icmp slt i32 %7, %9
  br i1 %10, label %11, label %13

11:                                               ; preds = %2
  %12 = load i32*, i32** %5, align 8
  store i32* %12, i32** %3, align 8
  br label %15

13:                                               ; preds = %2
  %14 = load i32*, i32** %4, align 8
  store i32* %14, i32** %3, align 8
  br label %15

15:                                               ; preds = %13, %11
  %16 = load i32*, i32** %3, align 8
  ret i32* %16
}

; Function Attrs: alwaysinline nounwind uwtable
define internal dereferenceable(4) i32* @_ZSt3maxIiERKT_S2_S2_(i32* dereferenceable(4) %0, i32* dereferenceable(4) %1) #0 {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca i32*, align 8
  store i32* %0, i32** %4, align 8
  store i32* %1, i32** %5, align 8
  %6 = load i32*, i32** %4, align 8
  %7 = load i32, i32* %6, align 4
  %8 = load i32*, i32** %5, align 8
  %9 = load i32, i32* %8, align 4
  %10 = icmp slt i32 %7, %9
  br i1 %10, label %11, label %13

11:                                               ; preds = %2
  %12 = load i32*, i32** %5, align 8
  store i32* %12, i32** %3, align 8
  br label %15

13:                                               ; preds = %2
  %14 = load i32*, i32** %4, align 8
  store i32* %14, i32** %3, align 8
  br label %15

15:                                               ; preds = %13, %11
  %16 = load i32*, i32** %3, align 8
  ret i32* %16
}

; Function Attrs: alwaysinline nounwind
declare i8* @llvm.stacksave() #2

; Function Attrs: alwaysinline nounwind
declare void @llvm.stackrestore(i8*) #2

; Function Attrs: alwaysinline nounwind uwtable
define internal void @cpu_parallel_range_for_task(i8* %0, i32 %1, i32 %2) #0 {
  %4 = alloca i8*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca %struct.range_task_helper_context, align 8
  %8 = alloca i8*, align 8
  %9 = alloca i64, align 8
  %10 = alloca i8*, align 8
  %11 = alloca %struct.RuntimeContext, align 8
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca i32, align 4
  store i8* %0, i8** %4, align 8
  store i32 %1, i32* %5, align 4
  store i32 %2, i32* %6, align 4
  %20 = load i8*, i8** %4, align 8
  %21 = bitcast i8* %20 to %struct.range_task_helper_context*
  %22 = bitcast %struct.range_task_helper_context* %7 to i8*
  %23 = bitcast %struct.range_task_helper_context* %21 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %22, i8* align 8 %23, i64 56, i1 false)
  %24 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 4
  %25 = load i64, i64* %24, align 8
  %26 = call i8* @llvm.stacksave()
  store i8* %26, i8** %8, align 8
  %27 = alloca i8, i64 %25, align 8
  store i64 %25, i64* %9, align 8
  %28 = getelementptr inbounds i8, i8* %27, i64 0
  store i8* %28, i8** %10, align 8
  %29 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 1
  %30 = load void (%struct.RuntimeContext*, i8*)*, void (%struct.RuntimeContext*, i8*)** %29, align 8
  %31 = icmp ne void (%struct.RuntimeContext*, i8*)* %30, null
  br i1 %31, label %32, label %38

32:                                               ; preds = %3
  %33 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 1
  %34 = load void (%struct.RuntimeContext*, i8*)*, void (%struct.RuntimeContext*, i8*)** %33, align 8
  %35 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 0
  %36 = load %struct.RuntimeContext*, %struct.RuntimeContext** %35, align 8
  %37 = load i8*, i8** %10, align 8
  call void %34(%struct.RuntimeContext* %36, i8* %37)
  br label %38

38:                                               ; preds = %32, %3
  %39 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 0
  %40 = load %struct.RuntimeContext*, %struct.RuntimeContext** %39, align 8
  %41 = bitcast %struct.RuntimeContext* %11 to i8*
  %42 = bitcast %struct.RuntimeContext* %40 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %41, i8* align 8 %42, i64 1624, i1 false)
  %43 = load i32, i32* %5, align 4
  %44 = getelementptr inbounds %struct.RuntimeContext, %struct.RuntimeContext* %11, i32 0, i32 3
  store i32 %43, i32* %44, align 8
  %45 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 8
  %46 = load i32, i32* %45, align 4
  %47 = icmp eq i32 %46, 1
  br i1 %47, label %48, label %77

48:                                               ; preds = %38
  %49 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 5
  %50 = load i32, i32* %49, align 8
  %51 = load i32, i32* %6, align 4
  %52 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 7
  %53 = load i32, i32* %52, align 8
  %54 = mul nsw i32 %51, %53
  %55 = add nsw i32 %50, %54
  store i32 %55, i32* %12, align 4
  %56 = load i32, i32* %12, align 4
  %57 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 7
  %58 = load i32, i32* %57, align 8
  %59 = add nsw i32 %56, %58
  store i32 %59, i32* %14, align 4
  %60 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 6
  %61 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4) %14, i32* dereferenceable(4) %60)
  %62 = load i32, i32* %61, align 4
  store i32 %62, i32* %13, align 4
  %63 = load i32, i32* %12, align 4
  store i32 %63, i32* %15, align 4
  br label %64

64:                                               ; preds = %73, %48
  %65 = load i32, i32* %15, align 4
  %66 = load i32, i32* %13, align 4
  %67 = icmp slt i32 %65, %66
  br i1 %67, label %68, label %76

68:                                               ; preds = %64
  %69 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 2
  %70 = load void (%struct.RuntimeContext*, i8*, i32)*, void (%struct.RuntimeContext*, i8*, i32)** %69, align 8
  %71 = load i8*, i8** %10, align 8
  %72 = load i32, i32* %15, align 4
  call void %70(%struct.RuntimeContext* %11, i8* %71, i32 %72)
  br label %73

73:                                               ; preds = %68
  %74 = load i32, i32* %15, align 4
  %75 = add nsw i32 %74, 1
  store i32 %75, i32* %15, align 4
  br label %64

76:                                               ; preds = %64
  br label %112

77:                                               ; preds = %38
  %78 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 8
  %79 = load i32, i32* %78, align 4
  %80 = icmp eq i32 %79, -1
  br i1 %80, label %81, label %111

81:                                               ; preds = %77
  %82 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 6
  %83 = load i32, i32* %82, align 4
  %84 = load i32, i32* %6, align 4
  %85 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 7
  %86 = load i32, i32* %85, align 8
  %87 = mul nsw i32 %84, %86
  %88 = sub nsw i32 %83, %87
  store i32 %88, i32* %16, align 4
  %89 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 5
  %90 = load i32, i32* %16, align 4
  %91 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 7
  %92 = load i32, i32* %91, align 8
  %93 = mul nsw i32 %90, %92
  store i32 %93, i32* %18, align 4
  %94 = call dereferenceable(4) i32* @_ZSt3maxIiERKT_S2_S2_(i32* dereferenceable(4) %89, i32* dereferenceable(4) %18)
  %95 = load i32, i32* %94, align 4
  store i32 %95, i32* %17, align 4
  %96 = load i32, i32* %16, align 4
  %97 = sub nsw i32 %96, 1
  store i32 %97, i32* %19, align 4
  br label %98

98:                                               ; preds = %107, %81
  %99 = load i32, i32* %19, align 4
  %100 = load i32, i32* %17, align 4
  %101 = icmp sge i32 %99, %100
  br i1 %101, label %102, label %110

102:                                              ; preds = %98
  %103 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 2
  %104 = load void (%struct.RuntimeContext*, i8*, i32)*, void (%struct.RuntimeContext*, i8*, i32)** %103, align 8
  %105 = load i8*, i8** %10, align 8
  %106 = load i32, i32* %19, align 4
  call void %104(%struct.RuntimeContext* %11, i8* %105, i32 %106)
  br label %107

107:                                              ; preds = %102
  %108 = load i32, i32* %19, align 4
  %109 = add nsw i32 %108, -1
  store i32 %109, i32* %19, align 4
  br label %98

110:                                              ; preds = %98
  br label %111

111:                                              ; preds = %110, %77
  br label %112

112:                                              ; preds = %111, %76
  %113 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 3
  %114 = load void (%struct.RuntimeContext*, i8*)*, void (%struct.RuntimeContext*, i8*)** %113, align 8
  %115 = icmp ne void (%struct.RuntimeContext*, i8*)* %114, null
  br i1 %115, label %116, label %122

116:                                              ; preds = %112
  %117 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 3
  %118 = load void (%struct.RuntimeContext*, i8*)*, void (%struct.RuntimeContext*, i8*)** %117, align 8
  %119 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %7, i32 0, i32 0
  %120 = load %struct.RuntimeContext*, %struct.RuntimeContext** %119, align 8
  %121 = load i8*, i8** %10, align 8
  call void %118(%struct.RuntimeContext* %120, i8* %121)
  br label %122

122:                                              ; preds = %116, %112
  %123 = load i8*, i8** %8, align 8
  call void @llvm.stackrestore(i8* %123)
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define internal void @cpu_parallel_range_for(%struct.RuntimeContext* %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, void (%struct.RuntimeContext*, i8*)* %6, void (%struct.RuntimeContext*, i8*, i32)* %7, void (%struct.RuntimeContext*, i8*)* %8, i64 %9) #0 {
  %11 = alloca %struct.RuntimeContext*, align 8
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca void (%struct.RuntimeContext*, i8*)*, align 8
  %18 = alloca void (%struct.RuntimeContext*, i8*, i32)*, align 8
  %19 = alloca void (%struct.RuntimeContext*, i8*)*, align 8
  %20 = alloca i64, align 8
  %21 = alloca %struct.range_task_helper_context, align 8
  %22 = alloca i32, align 4
  %23 = alloca i32, align 4
  %24 = alloca i32, align 4
  %25 = alloca i32, align 4
  %26 = alloca %struct.LLVMRuntime*, align 8
  store %struct.RuntimeContext* %0, %struct.RuntimeContext** %11, align 8
  store i32 %1, i32* %12, align 4
  store i32 %2, i32* %13, align 4
  store i32 %3, i32* %14, align 4
  store i32 %4, i32* %15, align 4
  store i32 %5, i32* %16, align 4
  store void (%struct.RuntimeContext*, i8*)* %6, void (%struct.RuntimeContext*, i8*)** %17, align 8
  store void (%struct.RuntimeContext*, i8*, i32)* %7, void (%struct.RuntimeContext*, i8*, i32)** %18, align 8
  store void (%struct.RuntimeContext*, i8*)* %8, void (%struct.RuntimeContext*, i8*)** %19, align 8
  store i64 %9, i64* %20, align 8
  call void @_ZN25range_task_helper_contextC2Ev(%struct.range_task_helper_context* %21) #5
  %27 = load %struct.RuntimeContext*, %struct.RuntimeContext** %11, align 8
  %28 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 0
  store %struct.RuntimeContext* %27, %struct.RuntimeContext** %28, align 8
  %29 = load void (%struct.RuntimeContext*, i8*)*, void (%struct.RuntimeContext*, i8*)** %17, align 8
  %30 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 1
  store void (%struct.RuntimeContext*, i8*)* %29, void (%struct.RuntimeContext*, i8*)** %30, align 8
  %31 = load i64, i64* %20, align 8
  %32 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 4
  store i64 %31, i64* %32, align 8
  %33 = load void (%struct.RuntimeContext*, i8*, i32)*, void (%struct.RuntimeContext*, i8*, i32)** %18, align 8
  %34 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 2
  store void (%struct.RuntimeContext*, i8*, i32)* %33, void (%struct.RuntimeContext*, i8*, i32)** %34, align 8
  %35 = load void (%struct.RuntimeContext*, i8*)*, void (%struct.RuntimeContext*, i8*)** %19, align 8
  %36 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 3
  store void (%struct.RuntimeContext*, i8*)* %35, void (%struct.RuntimeContext*, i8*)** %36, align 8
  %37 = load i32, i32* %13, align 4
  %38 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 5
  store i32 %37, i32* %38, align 8
  %39 = load i32, i32* %14, align 4
  %40 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 6
  store i32 %39, i32* %40, align 4
  %41 = load i32, i32* %15, align 4
  %42 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 8
  store i32 %41, i32* %42, align 4
  %43 = load i32, i32* %15, align 4
  %44 = icmp ne i32 %43, 1
  br i1 %44, label %45, label %52

45:                                               ; preds = %10
  %46 = load i32, i32* %15, align 4
  %47 = icmp ne i32 %46, -1
  br i1 %47, label %48, label %52

48:                                               ; preds = %45
  %49 = load %struct.RuntimeContext*, %struct.RuntimeContext** %11, align 8
  %50 = getelementptr inbounds %struct.RuntimeContext, %struct.RuntimeContext* %49, i32 0, i32 0
  %51 = load %struct.LLVMRuntime*, %struct.LLVMRuntime** %50, align 8
  call void @_Z13taichi_printfIJRiEEvP11LLVMRuntimePKcDpOT_(%struct.LLVMRuntime* %51, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.2, i64 0, i64 0), i32* dereferenceable(4) %15)
  call void @exit(i32 -1) #6
  unreachable

52:                                               ; preds = %45, %10
  %53 = load i32, i32* %16, align 4
  %54 = icmp eq i32 %53, 0
  br i1 %54, label %55, label %71

55:                                               ; preds = %52
  %56 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 6
  %57 = load i32, i32* %56, align 4
  %58 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 5
  %59 = load i32, i32* %58, align 8
  %60 = sub nsw i32 %57, %59
  %61 = load i32, i32* %15, align 4
  %62 = call i32 @abs(i32 %61) #7
  %63 = sdiv i32 %60, %62
  store i32 %63, i32* %22, align 4
  store i32 512, i32* %23, align 4
  store i32 1, i32* %24, align 4
  %64 = load i32, i32* %22, align 4
  %65 = load i32, i32* %12, align 4
  %66 = mul nsw i32 %65, 32
  %67 = sdiv i32 %64, %66
  store i32 %67, i32* %25, align 4
  %68 = call dereferenceable(4) i32* @_ZSt3maxIiERKT_S2_S2_(i32* dereferenceable(4) %24, i32* dereferenceable(4) %25)
  %69 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4) %23, i32* dereferenceable(4) %68)
  %70 = load i32, i32* %69, align 4
  store i32 %70, i32* %16, align 4
  br label %71

71:                                               ; preds = %55, %52
  %72 = load i32, i32* %16, align 4
  %73 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %21, i32 0, i32 7
  store i32 %72, i32* %73, align 8
  %74 = load %struct.RuntimeContext*, %struct.RuntimeContext** %11, align 8
  %75 = getelementptr inbounds %struct.RuntimeContext, %struct.RuntimeContext* %74, i32 0, i32 0
  %76 = load %struct.LLVMRuntime*, %struct.LLVMRuntime** %75, align 8
  store %struct.LLVMRuntime* %76, %struct.LLVMRuntime** %26, align 8
  %77 = load %struct.LLVMRuntime*, %struct.LLVMRuntime** %26, align 8
  %78 = getelementptr inbounds %struct.LLVMRuntime, %struct.LLVMRuntime* %77, i32 0, i32 12
  %79 = load void (i8*, i32, i32, i8*, void (i8*, i32, i32)*)*, void (i8*, i32, i32, i8*, void (i8*, i32, i32)*)** %78, align 8
  %80 = load %struct.LLVMRuntime*, %struct.LLVMRuntime** %26, align 8
  %81 = getelementptr inbounds %struct.LLVMRuntime, %struct.LLVMRuntime* %80, i32 0, i32 11
  %82 = load i8*, i8** %81, align 8
  %83 = load i32, i32* %14, align 4
  %84 = load i32, i32* %13, align 4
  %85 = sub nsw i32 %83, %84
  %86 = load i32, i32* %16, align 4
  %87 = add nsw i32 %85, %86
  %88 = sub nsw i32 %87, 1
  %89 = load i32, i32* %16, align 4
  %90 = sdiv i32 %88, %89
  %91 = load i32, i32* %12, align 4
  %92 = bitcast %struct.range_task_helper_context* %21 to i8*
  call void %79(i8* %82, i32 %90, i32 %91, i8* %92, void (i8*, i32, i32)* @cpu_parallel_range_for_task)
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define internal void @_ZN25range_task_helper_contextC2Ev(%struct.range_task_helper_context* %0) unnamed_addr #0 align 2 {
  %2 = alloca %struct.range_task_helper_context*, align 8
  store %struct.range_task_helper_context* %0, %struct.range_task_helper_context** %2, align 8
  %3 = load %struct.range_task_helper_context*, %struct.range_task_helper_context** %2, align 8
  %4 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %3, i32 0, i32 1
  store void (%struct.RuntimeContext*, i8*)* null, void (%struct.RuntimeContext*, i8*)** %4, align 8
  %5 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %3, i32 0, i32 2
  store void (%struct.RuntimeContext*, i8*, i32)* null, void (%struct.RuntimeContext*, i8*, i32)** %5, align 8
  %6 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %3, i32 0, i32 3
  store void (%struct.RuntimeContext*, i8*)* null, void (%struct.RuntimeContext*, i8*)** %6, align 8
  %7 = getelementptr inbounds %struct.range_task_helper_context, %struct.range_task_helper_context* %3, i32 0, i32 4
  store i64 1, i64* %7, align 8
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define internal void @_Z13taichi_printfIJRiEEvP11LLVMRuntimePKcDpOT_(%struct.LLVMRuntime* %0, i8* %1, i32* dereferenceable(4) %2) #0 {
  %4 = alloca %struct.LLVMRuntime*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i32*, align 8
  store %struct.LLVMRuntime* %0, %struct.LLVMRuntime** %4, align 8
  store i8* %1, i8** %5, align 8
  store i32* %2, i32** %6, align 8
  %7 = load %struct.LLVMRuntime*, %struct.LLVMRuntime** %4, align 8
  %8 = getelementptr inbounds %struct.LLVMRuntime, %struct.LLVMRuntime* %7, i32 0, i32 6
  %9 = load void (i8*, ...)*, void (i8*, ...)** %8, align 8
  %10 = load i8*, i8** %5, align 8
  %11 = load i32*, i32** %6, align 8
  %12 = load i32, i32* %11, align 4
  call void (i8*, ...) %9(i8* %10, i32 %12)
  ret void
}

; Function Attrs: alwaysinline noreturn nounwind
declare dso_local void @exit(i32) #3

; Function Attrs: alwaysinline nounwind readnone
declare dso_local i32 @abs(i32) #4

define void @run0(%struct.RuntimeContext* %context) {
entry:
  br label %body

final:                                            ; preds = %body
  ret void

body:                                             ; preds = %entry
  %0 = call i32 @RuntimeContext_get_extra_args(%struct.RuntimeContext* %context, i32 1, i32 0)
  %1 = call %struct.LLVMRuntime* @RuntimeContext_get_runtime(%struct.RuntimeContext* %context)
  %2 = call i8* @get_temporary_pointer(%struct.LLVMRuntime* %1, i64 0)
  %3 = bitcast i8* %2 to i32*
  store i32 %0, i32* %3
  br label %final
}

define void @run1(%struct.RuntimeContext* %0) {
entry:
  br label %body

final:                                            ; preds = %body
  ret void

body:                                             ; preds = %entry
  %1 = call %struct.LLVMRuntime* @RuntimeContext_get_runtime(%struct.RuntimeContext* %0)
  %2 = call i8* @get_temporary_pointer(%struct.LLVMRuntime* %1, i64 0)
  %3 = bitcast i8* %2 to i32*
  %4 = load i32, i32* %3
  call void @cpu_parallel_range_for(%struct.RuntimeContext* %0, i32 16, i32 0, i32 %4, i32 1, i32 32, void (%struct.RuntimeContext*, i8*)* null, void (%struct.RuntimeContext*, i8*, i32)* @function_body, void (%struct.RuntimeContext*, i8*)* null, i64 1)
  br label %final
}

define internal void @function_body(%struct.RuntimeContext* %0, i8* %1, i32 %2) {
allocs:
  %3 = alloca i32
  br label %entry

final:                                            ; preds = %function_body
  ret void

entry:                                            ; preds = %allocs
  br label %function_body

function_body:                                    ; preds = %entry
  store i32 %2, i32* %3
  %4 = load i32, i32* %3
  %5 = call %struct.LLVMRuntime* @RuntimeContext_get_runtime(%struct.RuntimeContext* %0)
  %6 = call i8* @get_temporary_pointer(%struct.LLVMRuntime* %5, i64 0)
  %7 = bitcast i8* %6 to i32*
  %8 = load i32, i32* %7
  %9 = srem i32 %4, %8
  %10 = call i64 @RuntimeContext_get_args(%struct.RuntimeContext* %0, i32 0)
  %11 = trunc i64 %10 to i32
  %12 = add i32 %11, %9
  %13 = call i64 @RuntimeContext_get_args(%struct.RuntimeContext* %0, i32 1)
  %14 = inttoptr i64 %13 to i32*
  %15 = call i32 @RuntimeContext_get_extra_args(%struct.RuntimeContext* %0, i32 1, i32 0)
  %16 = mul i32 0, %15
  %17 = add i32 %16, %9
  %18 = getelementptr i32, i32* %14, i32 %17
  store i32 %12, i32* %18
  br label %final
}

attributes #0 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline argmemonly nounwind willreturn }
attributes #2 = { alwaysinline nounwind }
attributes #3 = { alwaysinline noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { alwaysinline nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0-4ubuntu1 "}
