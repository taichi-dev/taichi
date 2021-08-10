// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef EIGEN_TEST_PART_1
#define EIGEN_UNALIGNED_VECTORIZE 1
#endif

#ifdef EIGEN_TEST_PART_2
#define EIGEN_UNALIGNED_VECTORIZE 0
#endif

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#undef EIGEN_DEFAULT_TO_ROW_MAJOR
#endif
#define EIGEN_DEBUG_ASSIGN
#include "main.h"
#include <typeinfo>

// Disable "ignoring attributes on template argument"
// for packet_traits<Packet*>
// => The only workaround would be to wrap _m128 and the likes
//    within wrappers.
#if EIGEN_GNUC_AT_LEAST(6,0)
    #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using internal::demangle_flags;
using internal::demangle_traversal;
using internal::demangle_unrolling;

template<typename Dst, typename Src>
bool test_assign(const Dst&, const Src&, int traversal, int unrolling)
{
  typedef internal::copy_using_evaluator_traits<internal::evaluator<Dst>,internal::evaluator<Src>, internal::assign_op<typename Dst::Scalar,typename Src::Scalar> > traits;
  bool res = traits::Traversal==traversal;
  if(unrolling==InnerUnrolling+CompleteUnrolling)
    res = res && (int(traits::Unrolling)==InnerUnrolling || int(traits::Unrolling)==CompleteUnrolling);
  else
    res = res && int(traits::Unrolling)==unrolling;
  if(!res)
  {
    std::cerr << "Src: " << demangle_flags(Src::Flags) << std::endl;
    std::cerr << "     " << demangle_flags(internal::evaluator<Src>::Flags) << std::endl;
    std::cerr << "Dst: " << demangle_flags(Dst::Flags) << std::endl;
    std::cerr << "     " << demangle_flags(internal::evaluator<Dst>::Flags) << std::endl;
    traits::debug();
    std::cerr << " Expected Traversal == " << demangle_traversal(traversal)
              << " got " << demangle_traversal(traits::Traversal) << "\n";
    std::cerr << " Expected Unrolling == " << demangle_unrolling(unrolling)
              << " got " << demangle_unrolling(traits::Unrolling) << "\n";
  }
  return res;
}

template<typename Dst, typename Src>
bool test_assign(int traversal, int unrolling)
{
  typedef internal::copy_using_evaluator_traits<internal::evaluator<Dst>,internal::evaluator<Src>, internal::assign_op<typename Dst::Scalar,typename Src::Scalar> > traits;
  bool res = traits::Traversal==traversal && traits::Unrolling==unrolling;
  if(!res)
  {
    std::cerr << "Src: " << demangle_flags(Src::Flags) << std::endl;
    std::cerr << "     " << demangle_flags(internal::evaluator<Src>::Flags) << std::endl;
    std::cerr << "Dst: " << demangle_flags(Dst::Flags) << std::endl;
    std::cerr << "     " << demangle_flags(internal::evaluator<Dst>::Flags) << std::endl;
    traits::debug();
    std::cerr << " Expected Traversal == " << demangle_traversal(traversal)
              << " got " << demangle_traversal(traits::Traversal) << "\n";
    std::cerr << " Expected Unrolling == " << demangle_unrolling(unrolling)
              << " got " << demangle_unrolling(traits::Unrolling) << "\n";
  }
  return res;
}

template<typename Xpr>
bool test_redux(const Xpr&, int traversal, int unrolling)
{
  typedef typename Xpr::Scalar Scalar;
  typedef internal::redux_traits<internal::scalar_sum_op<Scalar,Scalar>,internal::redux_evaluator<Xpr> > traits;
  
  bool res = traits::Traversal==traversal && traits::Unrolling==unrolling;
  if(!res)
  {
    std::cerr << demangle_flags(Xpr::Flags) << std::endl;
    std::cerr << demangle_flags(internal::evaluator<Xpr>::Flags) << std::endl;
    traits::debug();
    
    std::cerr << " Expected Traversal == " << demangle_traversal(traversal)
              << " got " << demangle_traversal(traits::Traversal) << "\n";
    std::cerr << " Expected Unrolling == " << demangle_unrolling(unrolling)
              << " got " << demangle_unrolling(traits::Unrolling) << "\n";
  }
  return res;
}

template<typename Scalar, bool Enable = internal::packet_traits<Scalar>::Vectorizable>
struct vectorization_logic
{
  typedef internal::packet_traits<Scalar> PacketTraits;
  
  typedef typename internal::packet_traits<Scalar>::type PacketType;
  typedef typename internal::unpacket_traits<PacketType>::half HalfPacketType;
  enum {
    PacketSize = internal::unpacket_traits<PacketType>::size,
    HalfPacketSize = internal::unpacket_traits<HalfPacketType>::size
  };
  static void run()
  {
    
    typedef Matrix<Scalar,PacketSize,1> Vector1;
    typedef Matrix<Scalar,Dynamic,1> VectorX;
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixXX;
    typedef Matrix<Scalar,PacketSize,PacketSize> Matrix11;
    typedef Matrix<Scalar,2*PacketSize,2*PacketSize> Matrix22;
    typedef Matrix<Scalar,(Matrix11::Flags&RowMajorBit)?16:4*PacketSize,(Matrix11::Flags&RowMajorBit)?4*PacketSize:16> Matrix44;
    typedef Matrix<Scalar,(Matrix11::Flags&RowMajorBit)?16:4*PacketSize,(Matrix11::Flags&RowMajorBit)?4*PacketSize:16,DontAlign|EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION> Matrix44u;
    typedef Matrix<Scalar,4*PacketSize,4*PacketSize,ColMajor> Matrix44c;
    typedef Matrix<Scalar,4*PacketSize,4*PacketSize,RowMajor> Matrix44r;

    typedef Matrix<Scalar,
        (PacketSize==8 ? 4 : PacketSize==4 ? 2 : PacketSize==2 ? 1 : /*PacketSize==1 ?*/ 1),
        (PacketSize==8 ? 2 : PacketSize==4 ? 2 : PacketSize==2 ? 2 : /*PacketSize==1 ?*/ 1)
      > Matrix1;

    typedef Matrix<Scalar,
        (PacketSize==8 ? 4 : PacketSize==4 ? 2 : PacketSize==2 ? 1 : /*PacketSize==1 ?*/ 1),
        (PacketSize==8 ? 2 : PacketSize==4 ? 2 : PacketSize==2 ? 2 : /*PacketSize==1 ?*/ 1),
      DontAlign|((Matrix1::Flags&RowMajorBit)?RowMajor:ColMajor)> Matrix1u;

    // this type is made such that it can only be vectorized when viewed as a linear 1D vector
    typedef Matrix<Scalar,
        (PacketSize==8 ? 4 : PacketSize==4 ? 6 : PacketSize==2 ? ((Matrix11::Flags&RowMajorBit)?2:3) : /*PacketSize==1 ?*/ 1),
        (PacketSize==8 ? 6 : PacketSize==4 ? 2 : PacketSize==2 ? ((Matrix11::Flags&RowMajorBit)?3:2) : /*PacketSize==1 ?*/ 3)
      > Matrix3;
    
    #if !EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT
    VERIFY(test_assign(Vector1(),Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1()+Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1().cwiseProduct(Vector1()),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1().template cast<Scalar>(),
      InnerVectorizedTraversal,CompleteUnrolling));


    VERIFY(test_assign(Vector1(),Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1()+Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1().cwiseProduct(Vector1()),
      InnerVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_assign(Matrix44(),Matrix44()+Matrix44(),
      InnerVectorizedTraversal,InnerUnrolling));

    VERIFY(test_assign(Matrix44u(),Matrix44()+Matrix44(),
      EIGEN_UNALIGNED_VECTORIZE ? InnerVectorizedTraversal : LinearTraversal,
      EIGEN_UNALIGNED_VECTORIZE ? InnerUnrolling : NoUnrolling));

    VERIFY(test_assign(Matrix1(),Matrix1()+Matrix1(),
      (Matrix1::InnerSizeAtCompileTime % PacketSize)==0 ? InnerVectorizedTraversal : LinearVectorizedTraversal,
      CompleteUnrolling));

    VERIFY(test_assign(Matrix1u(),Matrix1()+Matrix1(),
      EIGEN_UNALIGNED_VECTORIZE ? ((Matrix1::InnerSizeAtCompileTime % PacketSize)==0 ? InnerVectorizedTraversal : LinearVectorizedTraversal)
                                : LinearTraversal, CompleteUnrolling));

    VERIFY(test_assign(Matrix44c().col(1),Matrix44c().col(2)+Matrix44c().col(3),
      InnerVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_assign(Matrix44r().row(2),Matrix44r().row(1)+Matrix44r().row(1),
      InnerVectorizedTraversal,CompleteUnrolling));

    if(PacketSize>1)
    {
      typedef Matrix<Scalar,3,3,ColMajor> Matrix33c;
      typedef Matrix<Scalar,3,1,ColMajor> Vector3;
      VERIFY(test_assign(Matrix33c().row(2),Matrix33c().row(1)+Matrix33c().row(1),
        LinearTraversal,CompleteUnrolling));
      VERIFY(test_assign(Vector3(),Vector3()+Vector3(),
        EIGEN_UNALIGNED_VECTORIZE ? (HalfPacketSize==1 ? InnerVectorizedTraversal : LinearVectorizedTraversal) : (HalfPacketSize==1 ? InnerVectorizedTraversal : LinearTraversal), CompleteUnrolling));
      VERIFY(test_assign(Matrix33c().col(0),Matrix33c().col(1)+Matrix33c().col(1),
        EIGEN_UNALIGNED_VECTORIZE ? (HalfPacketSize==1 ? InnerVectorizedTraversal : LinearVectorizedTraversal) : (HalfPacketSize==1 ? SliceVectorizedTraversal : LinearTraversal),
        ((!EIGEN_UNALIGNED_VECTORIZE) && HalfPacketSize==1) ? NoUnrolling : CompleteUnrolling));

      VERIFY(test_assign(Matrix3(),Matrix3().cwiseProduct(Matrix3()),
        LinearVectorizedTraversal,CompleteUnrolling));

      VERIFY(test_assign(Matrix<Scalar,17,17>(),Matrix<Scalar,17,17>()+Matrix<Scalar,17,17>(),
        HalfPacketSize==1             ? InnerVectorizedTraversal  :
        EIGEN_UNALIGNED_VECTORIZE ? LinearVectorizedTraversal :
                                        LinearTraversal,
        NoUnrolling));

      VERIFY(test_assign(Matrix11(), Matrix11()+Matrix11(),InnerVectorizedTraversal,CompleteUnrolling));


      VERIFY(test_assign(Matrix11(),Matrix<Scalar,17,17>().template block<PacketSize,PacketSize>(2,3)+Matrix<Scalar,17,17>().template block<PacketSize,PacketSize>(8,4),
        (EIGEN_UNALIGNED_VECTORIZE) ? InnerVectorizedTraversal : DefaultTraversal, CompleteUnrolling|InnerUnrolling));

      VERIFY(test_assign(Vector1(),Matrix11()*Vector1(),
                         InnerVectorizedTraversal,CompleteUnrolling));

      VERIFY(test_assign(Matrix11(),Matrix11().lazyProduct(Matrix11()),
                         InnerVectorizedTraversal,InnerUnrolling+CompleteUnrolling));
    }

    VERIFY(test_redux(Vector1(),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Vector1().array()*Vector1().array(),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux((Vector1().array()*Vector1().array()).col(0),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix<Scalar,PacketSize,3>(),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix3(),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix44(),
      LinearVectorizedTraversal,NoUnrolling));

    VERIFY(test_redux(Matrix44().template block<(Matrix1::Flags&RowMajorBit)?4:PacketSize,(Matrix1::Flags&RowMajorBit)?PacketSize:4>(1,2),
      DefaultTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix44c().template block<2*PacketSize,1>(1,2),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix44r().template block<1,2*PacketSize>(2,1),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY((test_assign<
            Map<Matrix22, AlignedMax, OuterStride<3*PacketSize> >,
            Matrix22
            >(InnerVectorizedTraversal,CompleteUnrolling)));

    VERIFY((test_assign<
            Map<Matrix<Scalar,EIGEN_PLAIN_ENUM_MAX(2,PacketSize),EIGEN_PLAIN_ENUM_MAX(2,PacketSize)>, AlignedMax, InnerStride<3*PacketSize> >,
            Matrix<Scalar,EIGEN_PLAIN_ENUM_MAX(2,PacketSize),EIGEN_PLAIN_ENUM_MAX(2,PacketSize)>
            >(DefaultTraversal,PacketSize>=8?InnerUnrolling:CompleteUnrolling)));

    VERIFY((test_assign(Matrix11(), Matrix<Scalar,PacketSize,EIGEN_PLAIN_ENUM_MIN(2,PacketSize)>()*Matrix<Scalar,EIGEN_PLAIN_ENUM_MIN(2,PacketSize),PacketSize>(),
                        InnerVectorizedTraversal, CompleteUnrolling)));
    #endif

    VERIFY(test_assign(MatrixXX(10,10),MatrixXX(20,20).block(10,10,2,3),
      SliceVectorizedTraversal,NoUnrolling));

    VERIFY(test_redux(VectorX(10),
      LinearVectorizedTraversal,NoUnrolling));
  }
};

template<typename Scalar> struct vectorization_logic<Scalar,false>
{
  static void run() {}
};

template<typename Scalar, bool Enable = !internal::is_same<typename internal::unpacket_traits<typename internal::packet_traits<Scalar>::type>::half,
                                                           typename internal::packet_traits<Scalar>::type>::value >
struct vectorization_logic_half
{
  typedef internal::packet_traits<Scalar> PacketTraits;
  typedef typename internal::unpacket_traits<typename internal::packet_traits<Scalar>::type>::half PacketType;
  enum {
    PacketSize = internal::unpacket_traits<PacketType>::size
  };
  static void run()
  {
    
    typedef Matrix<Scalar,PacketSize,1> Vector1;
    typedef Matrix<Scalar,PacketSize,PacketSize> Matrix11;
    typedef Matrix<Scalar,5*PacketSize,7,ColMajor> Matrix57;
    typedef Matrix<Scalar,3*PacketSize,5,ColMajor> Matrix35;
    typedef Matrix<Scalar,5*PacketSize,7,DontAlign|ColMajor> Matrix57u;
//     typedef Matrix<Scalar,(Matrix11::Flags&RowMajorBit)?16:4*PacketSize,(Matrix11::Flags&RowMajorBit)?4*PacketSize:16> Matrix44;
//     typedef Matrix<Scalar,(Matrix11::Flags&RowMajorBit)?16:4*PacketSize,(Matrix11::Flags&RowMajorBit)?4*PacketSize:16,DontAlign|EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION> Matrix44u;
//     typedef Matrix<Scalar,4*PacketSize,4*PacketSize,ColMajor> Matrix44c;
//     typedef Matrix<Scalar,4*PacketSize,4*PacketSize,RowMajor> Matrix44r;

    typedef Matrix<Scalar,
        (PacketSize==8 ? 4 : PacketSize==4 ? 2 : PacketSize==2 ? 1 : /*PacketSize==1 ?*/ 1),
        (PacketSize==8 ? 2 : PacketSize==4 ? 2 : PacketSize==2 ? 2 : /*PacketSize==1 ?*/ 1)
      > Matrix1;

    typedef Matrix<Scalar,
        (PacketSize==8 ? 4 : PacketSize==4 ? 2 : PacketSize==2 ? 1 : /*PacketSize==1 ?*/ 1),
        (PacketSize==8 ? 2 : PacketSize==4 ? 2 : PacketSize==2 ? 2 : /*PacketSize==1 ?*/ 1),
      DontAlign|((Matrix1::Flags&RowMajorBit)?RowMajor:ColMajor)> Matrix1u;

    // this type is made such that it can only be vectorized when viewed as a linear 1D vector
    typedef Matrix<Scalar,
        (PacketSize==8 ? 4 : PacketSize==4 ? 6 : PacketSize==2 ? ((Matrix11::Flags&RowMajorBit)?2:3) : /*PacketSize==1 ?*/ 1),
        (PacketSize==8 ? 6 : PacketSize==4 ? 2 : PacketSize==2 ? ((Matrix11::Flags&RowMajorBit)?3:2) : /*PacketSize==1 ?*/ 3)
      > Matrix3;
    
    #if !EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT
    VERIFY(test_assign(Vector1(),Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1()+Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1().template segment<PacketSize>(0).derived(),
      EIGEN_UNALIGNED_VECTORIZE ? InnerVectorizedTraversal : LinearVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Scalar(2.1)*Vector1()-Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),(Scalar(2.1)*Vector1().template segment<PacketSize>(0)-Vector1().template segment<PacketSize>(0)).derived(),
      EIGEN_UNALIGNED_VECTORIZE ? InnerVectorizedTraversal : LinearVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1().cwiseProduct(Vector1()),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1().template cast<Scalar>(),
      InnerVectorizedTraversal,CompleteUnrolling));


    VERIFY(test_assign(Vector1(),Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1()+Vector1(),
      InnerVectorizedTraversal,CompleteUnrolling));
    VERIFY(test_assign(Vector1(),Vector1().cwiseProduct(Vector1()),
      InnerVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_assign(Matrix57(),Matrix57()+Matrix57(),
      InnerVectorizedTraversal,InnerUnrolling));

    VERIFY(test_assign(Matrix57u(),Matrix57()+Matrix57(),
      EIGEN_UNALIGNED_VECTORIZE ? InnerVectorizedTraversal : LinearTraversal,
      EIGEN_UNALIGNED_VECTORIZE ? InnerUnrolling : NoUnrolling));

    VERIFY(test_assign(Matrix1u(),Matrix1()+Matrix1(),
      EIGEN_UNALIGNED_VECTORIZE ? ((Matrix1::InnerSizeAtCompileTime % PacketSize)==0 ? InnerVectorizedTraversal : LinearVectorizedTraversal) : LinearTraversal,CompleteUnrolling));
        
    if(PacketSize>1)
    {
      typedef Matrix<Scalar,3,3,ColMajor> Matrix33c;
      VERIFY(test_assign(Matrix33c().row(2),Matrix33c().row(1)+Matrix33c().row(1),
        LinearTraversal,CompleteUnrolling));
      VERIFY(test_assign(Matrix33c().col(0),Matrix33c().col(1)+Matrix33c().col(1),
        EIGEN_UNALIGNED_VECTORIZE ? (PacketSize==1 ? InnerVectorizedTraversal : LinearVectorizedTraversal) : LinearTraversal,CompleteUnrolling));
              
      VERIFY(test_assign(Matrix3(),Matrix3().cwiseQuotient(Matrix3()),
        PacketTraits::HasDiv ? LinearVectorizedTraversal : LinearTraversal,CompleteUnrolling));
        
      VERIFY(test_assign(Matrix<Scalar,17,17>(),Matrix<Scalar,17,17>()+Matrix<Scalar,17,17>(),
        EIGEN_UNALIGNED_VECTORIZE ? (PacketSize==1 ? InnerVectorizedTraversal : LinearVectorizedTraversal) : LinearTraversal,
        NoUnrolling));
        
      VERIFY(test_assign(Matrix11(),Matrix<Scalar,17,17>().template block<PacketSize,PacketSize>(2,3)+Matrix<Scalar,17,17>().template block<PacketSize,PacketSize>(8,4),
        EIGEN_UNALIGNED_VECTORIZE ? InnerVectorizedTraversal : DefaultTraversal,PacketSize>4?InnerUnrolling:CompleteUnrolling));

      VERIFY(test_assign(Vector1(),Matrix11()*Vector1(),
                         InnerVectorizedTraversal,CompleteUnrolling));

      VERIFY(test_assign(Matrix11(),Matrix11().lazyProduct(Matrix11()),
                         InnerVectorizedTraversal,InnerUnrolling+CompleteUnrolling));
    }
    
    VERIFY(test_redux(Vector1(),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix<Scalar,PacketSize,3>(),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix3(),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix35(),
      LinearVectorizedTraversal,CompleteUnrolling));

    VERIFY(test_redux(Matrix57().template block<PacketSize,3>(1,0),
      DefaultTraversal,CompleteUnrolling));

    VERIFY((test_assign<
            Map<Matrix<Scalar,EIGEN_PLAIN_ENUM_MAX(2,PacketSize),EIGEN_PLAIN_ENUM_MAX(2,PacketSize)>, AlignedMax, InnerStride<3*PacketSize> >,
            Matrix<Scalar,EIGEN_PLAIN_ENUM_MAX(2,PacketSize),EIGEN_PLAIN_ENUM_MAX(2,PacketSize)>
            >(DefaultTraversal,CompleteUnrolling)));

    VERIFY((test_assign(Matrix57(), Matrix<Scalar,5*PacketSize,3>()*Matrix<Scalar,3,7>(),
                        InnerVectorizedTraversal, InnerUnrolling|CompleteUnrolling)));
    #endif
  }
};

template<typename Scalar> struct vectorization_logic_half<Scalar,false>
{
  static void run() {}
};

void test_vectorization_logic()
{

#ifdef EIGEN_VECTORIZE

  CALL_SUBTEST( vectorization_logic<int>::run() );
  CALL_SUBTEST( vectorization_logic<float>::run() );
  CALL_SUBTEST( vectorization_logic<double>::run() );
  CALL_SUBTEST( vectorization_logic<std::complex<float> >::run() );
  CALL_SUBTEST( vectorization_logic<std::complex<double> >::run() );
  
  CALL_SUBTEST( vectorization_logic_half<int>::run() );
  CALL_SUBTEST( vectorization_logic_half<float>::run() );
  CALL_SUBTEST( vectorization_logic_half<double>::run() );
  CALL_SUBTEST( vectorization_logic_half<std::complex<float> >::run() );
  CALL_SUBTEST( vectorization_logic_half<std::complex<double> >::run() );
  
  if(internal::packet_traits<float>::Vectorizable)
  {
    VERIFY(test_assign(Matrix<float,3,3>(),Matrix<float,3,3>()+Matrix<float,3,3>(),
      EIGEN_UNALIGNED_VECTORIZE ? LinearVectorizedTraversal : LinearTraversal,CompleteUnrolling));
      
    VERIFY(test_redux(Matrix<float,5,2>(),
      EIGEN_UNALIGNED_VECTORIZE ? LinearVectorizedTraversal : DefaultTraversal,CompleteUnrolling));
  }
  
  if(internal::packet_traits<double>::Vectorizable)
  {
    VERIFY(test_assign(Matrix<double,3,3>(),Matrix<double,3,3>()+Matrix<double,3,3>(),
      EIGEN_UNALIGNED_VECTORIZE ? LinearVectorizedTraversal : LinearTraversal,CompleteUnrolling));
    
    VERIFY(test_redux(Matrix<double,7,3>(),
      EIGEN_UNALIGNED_VECTORIZE ? LinearVectorizedTraversal : DefaultTraversal,CompleteUnrolling));
  }
#endif // EIGEN_VECTORIZE

}
