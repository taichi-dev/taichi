// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include "unsupported/Eigen/SpecialFunctions"

#if defined __GNUC__ && __GNUC__>=6
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
// using namespace Eigen;

namespace Eigen {
namespace internal {
template<typename T> T negate(const T& x) { return -x; }
}
}

// NOTE: we disbale inlining for this function to workaround a GCC issue when using -O3 and the i387 FPU.
template<typename Scalar> EIGEN_DONT_INLINE
bool isApproxAbs(const Scalar& a, const Scalar& b, const typename NumTraits<Scalar>::Real& refvalue)
{
  return internal::isMuchSmallerThan(a-b, refvalue);
}

template<typename Scalar> bool areApproxAbs(const Scalar* a, const Scalar* b, int size, const typename NumTraits<Scalar>::Real& refvalue)
{
  for (int i=0; i<size; ++i)
  {
    if (!isApproxAbs(a[i],b[i],refvalue))
    {
      std::cout << "ref: [" << Map<const Matrix<Scalar,1,Dynamic> >(a,size) << "]" << " != vec: [" << Map<const Matrix<Scalar,1,Dynamic> >(b,size) << "]\n";
      return false;
    }
  }
  return true;
}

template<typename Scalar> bool areApprox(const Scalar* a, const Scalar* b, int size)
{
  for (int i=0; i<size; ++i)
  {
    if (a[i]!=b[i] && !internal::isApprox(a[i],b[i]))
    {
      std::cout << "ref: [" << Map<const Matrix<Scalar,1,Dynamic> >(a,size) << "]" << " != vec: [" << Map<const Matrix<Scalar,1,Dynamic> >(b,size) << "]\n";
      return false;
    }
  }
  return true;
}

#define CHECK_CWISE1(REFOP, POP) { \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i]); \
  internal::pstore(data2, POP(internal::pload<Packet>(data1))); \
  VERIFY(areApprox(ref, data2, PacketSize) && #POP); \
}

template<bool Cond,typename Packet>
struct packet_helper
{
  template<typename T>
  inline Packet load(const T* from) const { return internal::pload<Packet>(from); }

  template<typename T>
  inline void store(T* to, const Packet& x) const { internal::pstore(to,x); }
};

template<typename Packet>
struct packet_helper<false,Packet>
{
  template<typename T>
  inline T load(const T* from) const { return *from; }

  template<typename T>
  inline void store(T* to, const T& x) const { *to = x; }
};

#define CHECK_CWISE1_IF(COND, REFOP, POP) if(COND) { \
  packet_helper<COND,Packet> h; \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i]); \
  h.store(data2, POP(h.load(data1))); \
  VERIFY(areApprox(ref, data2, PacketSize) && #POP); \
}

#define CHECK_CWISE2_IF(COND, REFOP, POP) if(COND) { \
  packet_helper<COND,Packet> h; \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i], data1[i+PacketSize]); \
  h.store(data2, POP(h.load(data1),h.load(data1+PacketSize))); \
  VERIFY(areApprox(ref, data2, PacketSize) && #POP); \
}

#define REF_ADD(a,b) ((a)+(b))
#define REF_SUB(a,b) ((a)-(b))
#define REF_MUL(a,b) ((a)*(b))
#define REF_DIV(a,b) ((a)/(b))

template<typename Scalar> void packetmath()
{
  using std::abs;
  typedef internal::packet_traits<Scalar> PacketTraits;
  typedef typename PacketTraits::type Packet;
  const int PacketSize = PacketTraits::size;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  const int max_size = PacketSize > 4 ? PacketSize : 4;
  const int size = PacketSize*max_size;
  EIGEN_ALIGN_MAX Scalar data1[size];
  EIGEN_ALIGN_MAX Scalar data2[size];
  EIGEN_ALIGN_MAX Packet packets[PacketSize*2];
  EIGEN_ALIGN_MAX Scalar ref[size];
  RealScalar refvalue = 0;
  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>()/RealScalar(PacketSize);
    data2[i] = internal::random<Scalar>()/RealScalar(PacketSize);
    refvalue = (std::max)(refvalue,abs(data1[i]));
  }

  internal::pstore(data2, internal::pload<Packet>(data1));
  VERIFY(areApprox(data1, data2, PacketSize) && "aligned load/store");

  for (int offset=0; offset<PacketSize; ++offset)
  {
    internal::pstore(data2, internal::ploadu<Packet>(data1+offset));
    VERIFY(areApprox(data1+offset, data2, PacketSize) && "internal::ploadu");
  }

  for (int offset=0; offset<PacketSize; ++offset)
  {
    internal::pstoreu(data2+offset, internal::pload<Packet>(data1));
    VERIFY(areApprox(data1, data2+offset, PacketSize) && "internal::pstoreu");
  }

  for (int offset=0; offset<PacketSize; ++offset)
  {
    packets[0] = internal::pload<Packet>(data1);
    packets[1] = internal::pload<Packet>(data1+PacketSize);
         if (offset==0) internal::palign<0>(packets[0], packets[1]);
    else if (offset==1) internal::palign<1>(packets[0], packets[1]);
    else if (offset==2) internal::palign<2>(packets[0], packets[1]);
    else if (offset==3) internal::palign<3>(packets[0], packets[1]);
    else if (offset==4) internal::palign<4>(packets[0], packets[1]);
    else if (offset==5) internal::palign<5>(packets[0], packets[1]);
    else if (offset==6) internal::palign<6>(packets[0], packets[1]);
    else if (offset==7) internal::palign<7>(packets[0], packets[1]);
    else if (offset==8) internal::palign<8>(packets[0], packets[1]);
    else if (offset==9) internal::palign<9>(packets[0], packets[1]);
    else if (offset==10) internal::palign<10>(packets[0], packets[1]);
    else if (offset==11) internal::palign<11>(packets[0], packets[1]);
    else if (offset==12) internal::palign<12>(packets[0], packets[1]);
    else if (offset==13) internal::palign<13>(packets[0], packets[1]);
    else if (offset==14) internal::palign<14>(packets[0], packets[1]);
    else if (offset==15) internal::palign<15>(packets[0], packets[1]);
    internal::pstore(data2, packets[0]);

    for (int i=0; i<PacketSize; ++i)
      ref[i] = data1[i+offset];

    VERIFY(areApprox(ref, data2, PacketSize) && "internal::palign");
  }

  VERIFY((!PacketTraits::Vectorizable) || PacketTraits::HasAdd);
  VERIFY((!PacketTraits::Vectorizable) || PacketTraits::HasSub);
  VERIFY((!PacketTraits::Vectorizable) || PacketTraits::HasMul);
  VERIFY((!PacketTraits::Vectorizable) || PacketTraits::HasNegate);
  VERIFY((internal::is_same<Scalar,int>::value) || (!PacketTraits::Vectorizable) || PacketTraits::HasDiv);

  CHECK_CWISE2_IF(PacketTraits::HasAdd, REF_ADD,  internal::padd);
  CHECK_CWISE2_IF(PacketTraits::HasSub, REF_SUB,  internal::psub);
  CHECK_CWISE2_IF(PacketTraits::HasMul, REF_MUL,  internal::pmul);
  CHECK_CWISE2_IF(PacketTraits::HasDiv, REF_DIV, internal::pdiv);

  CHECK_CWISE1(internal::negate, internal::pnegate);
  CHECK_CWISE1(numext::conj, internal::pconj);

  for(int offset=0;offset<3;++offset)
  {
    for (int i=0; i<PacketSize; ++i)
      ref[i] = data1[offset];
    internal::pstore(data2, internal::pset1<Packet>(data1[offset]));
    VERIFY(areApprox(ref, data2, PacketSize) && "internal::pset1");
  }

  {
    for (int i=0; i<PacketSize*4; ++i)
      ref[i] = data1[i/PacketSize];
    Packet A0, A1, A2, A3;
    internal::pbroadcast4<Packet>(data1, A0, A1, A2, A3);
    internal::pstore(data2+0*PacketSize, A0);
    internal::pstore(data2+1*PacketSize, A1);
    internal::pstore(data2+2*PacketSize, A2);
    internal::pstore(data2+3*PacketSize, A3);
    VERIFY(areApprox(ref, data2, 4*PacketSize) && "internal::pbroadcast4");
  }

  {
    for (int i=0; i<PacketSize*2; ++i)
      ref[i] = data1[i/PacketSize];
    Packet A0, A1;
    internal::pbroadcast2<Packet>(data1, A0, A1);
    internal::pstore(data2+0*PacketSize, A0);
    internal::pstore(data2+1*PacketSize, A1);
    VERIFY(areApprox(ref, data2, 2*PacketSize) && "internal::pbroadcast2");
  }

  VERIFY(internal::isApprox(data1[0], internal::pfirst(internal::pload<Packet>(data1))) && "internal::pfirst");

  if(PacketSize>1)
  {
    for(int offset=0;offset<4;++offset)
    {
      for(int i=0;i<PacketSize/2;++i)
        ref[2*i+0] = ref[2*i+1] = data1[offset+i];
      internal::pstore(data2,internal::ploaddup<Packet>(data1+offset));
      VERIFY(areApprox(ref, data2, PacketSize) && "ploaddup");
    }
  }

  if(PacketSize>2)
  {
    for(int offset=0;offset<4;++offset)
    {
      for(int i=0;i<PacketSize/4;++i)
        ref[4*i+0] = ref[4*i+1] = ref[4*i+2] = ref[4*i+3] = data1[offset+i];
      internal::pstore(data2,internal::ploadquad<Packet>(data1+offset));
      VERIFY(areApprox(ref, data2, PacketSize) && "ploadquad");
    }
  }

  ref[0] = 0;
  for (int i=0; i<PacketSize; ++i)
    ref[0] += data1[i];
  VERIFY(isApproxAbs(ref[0], internal::predux(internal::pload<Packet>(data1)), refvalue) && "internal::predux");

  {
    int newsize = PacketSize>4?PacketSize/2:PacketSize;
    for (int i=0; i<newsize; ++i)
      ref[i] = 0;
    for (int i=0; i<PacketSize; ++i)
      ref[i%newsize] += data1[i];
    internal::pstore(data2, internal::predux_downto4(internal::pload<Packet>(data1)));
    VERIFY(areApprox(ref, data2, newsize) && "internal::predux_downto4");
  }

  ref[0] = 1;
  for (int i=0; i<PacketSize; ++i)
    ref[0] *= data1[i];
  VERIFY(internal::isApprox(ref[0], internal::predux_mul(internal::pload<Packet>(data1))) && "internal::predux_mul");

  for (int j=0; j<PacketSize; ++j)
  {
    ref[j] = 0;
    for (int i=0; i<PacketSize; ++i)
      ref[j] += data1[i+j*PacketSize];
    packets[j] = internal::pload<Packet>(data1+j*PacketSize);
  }
  internal::pstore(data2, internal::preduxp(packets));
  VERIFY(areApproxAbs(ref, data2, PacketSize, refvalue) && "internal::preduxp");

  for (int i=0; i<PacketSize; ++i)
    ref[i] = data1[PacketSize-i-1];
  internal::pstore(data2, internal::preverse(internal::pload<Packet>(data1)));
  VERIFY(areApprox(ref, data2, PacketSize) && "internal::preverse");

  internal::PacketBlock<Packet> kernel;
  for (int i=0; i<PacketSize; ++i) {
    kernel.packet[i] = internal::pload<Packet>(data1+i*PacketSize);
  }
  ptranspose(kernel);
  for (int i=0; i<PacketSize; ++i) {
    internal::pstore(data2, kernel.packet[i]);
    for (int j = 0; j < PacketSize; ++j) {
      VERIFY(isApproxAbs(data2[j], data1[i+j*PacketSize], refvalue) && "ptranspose");
    }
  }

  if (PacketTraits::HasBlend) {
    Packet thenPacket = internal::pload<Packet>(data1);
    Packet elsePacket = internal::pload<Packet>(data2);
    EIGEN_ALIGN_MAX internal::Selector<PacketSize> selector;
    for (int i = 0; i < PacketSize; ++i) {
      selector.select[i] = i;
    }

    Packet blend = internal::pblend(selector, thenPacket, elsePacket);
    EIGEN_ALIGN_MAX Scalar result[size];
    internal::pstore(result, blend);
    for (int i = 0; i < PacketSize; ++i) {
      VERIFY(isApproxAbs(result[i], (selector.select[i] ? data1[i] : data2[i]), refvalue));
    }
  }

  if (PacketTraits::HasBlend) {
    // pinsertfirst
    for (int i=0; i<PacketSize; ++i)
      ref[i] = data1[i];
    Scalar s = internal::random<Scalar>();
    ref[0] = s;
    internal::pstore(data2, internal::pinsertfirst(internal::pload<Packet>(data1),s));
    VERIFY(areApprox(ref, data2, PacketSize) && "internal::pinsertfirst");
  }

  if (PacketTraits::HasBlend) {
    // pinsertlast
    for (int i=0; i<PacketSize; ++i)
      ref[i] = data1[i];
    Scalar s = internal::random<Scalar>();
    ref[PacketSize-1] = s;
    internal::pstore(data2, internal::pinsertlast(internal::pload<Packet>(data1),s));
    VERIFY(areApprox(ref, data2, PacketSize) && "internal::pinsertlast");
  }
}

template<typename Scalar> void packetmath_real()
{
  using std::abs;
  typedef internal::packet_traits<Scalar> PacketTraits;
  typedef typename PacketTraits::type Packet;
  const int PacketSize = PacketTraits::size;

  const int size = PacketSize*4;
  EIGEN_ALIGN_MAX Scalar data1[PacketTraits::size*4];
  EIGEN_ALIGN_MAX Scalar data2[PacketTraits::size*4];
  EIGEN_ALIGN_MAX Scalar ref[PacketTraits::size*4];

  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(-1,1) * std::pow(Scalar(10), internal::random<Scalar>(-3,3));
    data2[i] = internal::random<Scalar>(-1,1) * std::pow(Scalar(10), internal::random<Scalar>(-3,3));
  }
  CHECK_CWISE1_IF(PacketTraits::HasSin, std::sin, internal::psin);
  CHECK_CWISE1_IF(PacketTraits::HasCos, std::cos, internal::pcos);
  CHECK_CWISE1_IF(PacketTraits::HasTan, std::tan, internal::ptan);

  CHECK_CWISE1_IF(PacketTraits::HasRound, numext::round, internal::pround);
  CHECK_CWISE1_IF(PacketTraits::HasCeil, numext::ceil, internal::pceil);
  CHECK_CWISE1_IF(PacketTraits::HasFloor, numext::floor, internal::pfloor);

  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(-1,1);
    data2[i] = internal::random<Scalar>(-1,1);
  }
  CHECK_CWISE1_IF(PacketTraits::HasASin, std::asin, internal::pasin);
  CHECK_CWISE1_IF(PacketTraits::HasACos, std::acos, internal::pacos);

  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(-87,88);
    data2[i] = internal::random<Scalar>(-87,88);
  }
  CHECK_CWISE1_IF(PacketTraits::HasExp, std::exp, internal::pexp);
  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(-1,1) * std::pow(Scalar(10), internal::random<Scalar>(-6,6));
    data2[i] = internal::random<Scalar>(-1,1) * std::pow(Scalar(10), internal::random<Scalar>(-6,6));
  }
  CHECK_CWISE1_IF(PacketTraits::HasTanh, std::tanh, internal::ptanh);
  if(PacketTraits::HasExp && PacketTraits::size>=2)
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    data1[1] = std::numeric_limits<Scalar>::epsilon();
    packet_helper<PacketTraits::HasExp,Packet> h;
    h.store(data2, internal::pexp(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
    VERIFY_IS_EQUAL(std::exp(std::numeric_limits<Scalar>::epsilon()), data2[1]);

    data1[0] = -std::numeric_limits<Scalar>::epsilon();
    data1[1] = 0;
    h.store(data2, internal::pexp(h.load(data1)));
    VERIFY_IS_EQUAL(std::exp(-std::numeric_limits<Scalar>::epsilon()), data2[0]);
    VERIFY_IS_EQUAL(std::exp(Scalar(0)), data2[1]);

    data1[0] = (std::numeric_limits<Scalar>::min)();
    data1[1] = -(std::numeric_limits<Scalar>::min)();
    h.store(data2, internal::pexp(h.load(data1)));
    VERIFY_IS_EQUAL(std::exp((std::numeric_limits<Scalar>::min)()), data2[0]);
    VERIFY_IS_EQUAL(std::exp(-(std::numeric_limits<Scalar>::min)()), data2[1]);

    data1[0] = std::numeric_limits<Scalar>::denorm_min();
    data1[1] = -std::numeric_limits<Scalar>::denorm_min();
    h.store(data2, internal::pexp(h.load(data1)));
    VERIFY_IS_EQUAL(std::exp(std::numeric_limits<Scalar>::denorm_min()), data2[0]);
    VERIFY_IS_EQUAL(std::exp(-std::numeric_limits<Scalar>::denorm_min()), data2[1]);
  }

  if (PacketTraits::HasTanh) {
    // NOTE this test migh fail with GCC prior to 6.3, see MathFunctionsImpl.h for details.
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    packet_helper<internal::packet_traits<Scalar>::HasTanh,Packet> h;
    h.store(data2, internal::ptanh(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }

#if EIGEN_HAS_C99_MATH
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    packet_helper<internal::packet_traits<Scalar>::HasLGamma,Packet> h;
    h.store(data2, internal::plgamma(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    packet_helper<internal::packet_traits<Scalar>::HasErf,Packet> h;
    h.store(data2, internal::perf(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    packet_helper<internal::packet_traits<Scalar>::HasErfc,Packet> h;
    h.store(data2, internal::perfc(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
#endif  // EIGEN_HAS_C99_MATH

  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(0,1) * std::pow(Scalar(10), internal::random<Scalar>(-6,6));
    data2[i] = internal::random<Scalar>(0,1) * std::pow(Scalar(10), internal::random<Scalar>(-6,6));
  }

  if(internal::random<float>(0,1)<0.1f)
    data1[internal::random<int>(0, PacketSize)] = 0;
  CHECK_CWISE1_IF(PacketTraits::HasSqrt, std::sqrt, internal::psqrt);
  CHECK_CWISE1_IF(PacketTraits::HasLog, std::log, internal::plog);
#if EIGEN_HAS_C99_MATH && (__cplusplus > 199711L)
  CHECK_CWISE1_IF(PacketTraits::HasLog1p, std::log1p, internal::plog1p);
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasLGamma, std::lgamma, internal::plgamma);
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasErf, std::erf, internal::perf);
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasErfc, std::erfc, internal::perfc);
#endif

  if(PacketTraits::HasLog && PacketTraits::size>=2)
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    data1[1] = std::numeric_limits<Scalar>::epsilon();
    packet_helper<PacketTraits::HasLog,Packet> h;
    h.store(data2, internal::plog(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
    VERIFY_IS_EQUAL(std::log(std::numeric_limits<Scalar>::epsilon()), data2[1]);

    data1[0] = -std::numeric_limits<Scalar>::epsilon();
    data1[1] = 0;
    h.store(data2, internal::plog(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
    VERIFY_IS_EQUAL(std::log(Scalar(0)), data2[1]);

    data1[0] = (std::numeric_limits<Scalar>::min)();
    data1[1] = -(std::numeric_limits<Scalar>::min)();
    h.store(data2, internal::plog(h.load(data1)));
    VERIFY_IS_EQUAL(std::log((std::numeric_limits<Scalar>::min)()), data2[0]);
    VERIFY((numext::isnan)(data2[1]));

    data1[0] = std::numeric_limits<Scalar>::denorm_min();
    data1[1] = -std::numeric_limits<Scalar>::denorm_min();
    h.store(data2, internal::plog(h.load(data1)));
    // VERIFY_IS_EQUAL(std::log(std::numeric_limits<Scalar>::denorm_min()), data2[0]);
    VERIFY((numext::isnan)(data2[1]));

    data1[0] = Scalar(-1.0f);
    h.store(data2, internal::plog(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
    h.store(data2, internal::psqrt(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
    VERIFY((numext::isnan)(data2[1]));
  }
}

template<typename Scalar> void packetmath_notcomplex()
{
  using std::abs;
  typedef internal::packet_traits<Scalar> PacketTraits;
  typedef typename PacketTraits::type Packet;
  const int PacketSize = PacketTraits::size;

  EIGEN_ALIGN_MAX Scalar data1[PacketTraits::size*4];
  EIGEN_ALIGN_MAX Scalar data2[PacketTraits::size*4];
  EIGEN_ALIGN_MAX Scalar ref[PacketTraits::size*4];

  Array<Scalar,Dynamic,1>::Map(data1, PacketTraits::size*4).setRandom();

  ref[0] = data1[0];
  for (int i=0; i<PacketSize; ++i)
    ref[0] = (std::min)(ref[0],data1[i]);
  VERIFY(internal::isApprox(ref[0], internal::predux_min(internal::pload<Packet>(data1))) && "internal::predux_min");

  VERIFY((!PacketTraits::Vectorizable) || PacketTraits::HasMin);
  VERIFY((!PacketTraits::Vectorizable) || PacketTraits::HasMax);

  CHECK_CWISE2_IF(PacketTraits::HasMin, (std::min), internal::pmin);
  CHECK_CWISE2_IF(PacketTraits::HasMax, (std::max), internal::pmax);
  CHECK_CWISE1(abs, internal::pabs);

  ref[0] = data1[0];
  for (int i=0; i<PacketSize; ++i)
    ref[0] = (std::max)(ref[0],data1[i]);
  VERIFY(internal::isApprox(ref[0], internal::predux_max(internal::pload<Packet>(data1))) && "internal::predux_max");

  for (int i=0; i<PacketSize; ++i)
    ref[i] = data1[0]+Scalar(i);
  internal::pstore(data2, internal::plset<Packet>(data1[0]));
  VERIFY(areApprox(ref, data2, PacketSize) && "internal::plset");
}

template<typename Scalar,bool ConjLhs,bool ConjRhs> void test_conj_helper(Scalar* data1, Scalar* data2, Scalar* ref, Scalar* pval)
{
  typedef internal::packet_traits<Scalar> PacketTraits;
  typedef typename PacketTraits::type Packet;
  const int PacketSize = PacketTraits::size;

  internal::conj_if<ConjLhs> cj0;
  internal::conj_if<ConjRhs> cj1;
  internal::conj_helper<Scalar,Scalar,ConjLhs,ConjRhs> cj;
  internal::conj_helper<Packet,Packet,ConjLhs,ConjRhs> pcj;

  for(int i=0;i<PacketSize;++i)
  {
    ref[i] = cj0(data1[i]) * cj1(data2[i]);
    VERIFY(internal::isApprox(ref[i], cj.pmul(data1[i],data2[i])) && "conj_helper pmul");
  }
  internal::pstore(pval,pcj.pmul(internal::pload<Packet>(data1),internal::pload<Packet>(data2)));
  VERIFY(areApprox(ref, pval, PacketSize) && "conj_helper pmul");

  for(int i=0;i<PacketSize;++i)
  {
    Scalar tmp = ref[i];
    ref[i] += cj0(data1[i]) * cj1(data2[i]);
    VERIFY(internal::isApprox(ref[i], cj.pmadd(data1[i],data2[i],tmp)) && "conj_helper pmadd");
  }
  internal::pstore(pval,pcj.pmadd(internal::pload<Packet>(data1),internal::pload<Packet>(data2),internal::pload<Packet>(pval)));
  VERIFY(areApprox(ref, pval, PacketSize) && "conj_helper pmadd");
}

template<typename Scalar> void packetmath_complex()
{
  typedef internal::packet_traits<Scalar> PacketTraits;
  typedef typename PacketTraits::type Packet;
  const int PacketSize = PacketTraits::size;

  const int size = PacketSize*4;
  EIGEN_ALIGN_MAX Scalar data1[PacketSize*4];
  EIGEN_ALIGN_MAX Scalar data2[PacketSize*4];
  EIGEN_ALIGN_MAX Scalar ref[PacketSize*4];
  EIGEN_ALIGN_MAX Scalar pval[PacketSize*4];

  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>() * Scalar(1e2);
    data2[i] = internal::random<Scalar>() * Scalar(1e2);
  }

  test_conj_helper<Scalar,false,false> (data1,data2,ref,pval);
  test_conj_helper<Scalar,false,true>  (data1,data2,ref,pval);
  test_conj_helper<Scalar,true,false>  (data1,data2,ref,pval);
  test_conj_helper<Scalar,true,true>   (data1,data2,ref,pval);

  {
    for(int i=0;i<PacketSize;++i)
      ref[i] = Scalar(std::imag(data1[i]),std::real(data1[i]));
    internal::pstore(pval,internal::pcplxflip(internal::pload<Packet>(data1)));
    VERIFY(areApprox(ref, pval, PacketSize) && "pcplxflip");
  }
}

template<typename Scalar> void packetmath_scatter_gather()
{
  typedef internal::packet_traits<Scalar> PacketTraits;
  typedef typename PacketTraits::type Packet;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const int PacketSize = PacketTraits::size;
  EIGEN_ALIGN_MAX Scalar data1[PacketSize];
  RealScalar refvalue = 0;
  for (int i=0; i<PacketSize; ++i) {
    data1[i] = internal::random<Scalar>()/RealScalar(PacketSize);
  }

  int stride = internal::random<int>(1,20);

  EIGEN_ALIGN_MAX Scalar buffer[PacketSize*20];
  memset(buffer, 0, 20*PacketSize*sizeof(Scalar));
  Packet packet = internal::pload<Packet>(data1);
  internal::pscatter<Scalar, Packet>(buffer, packet, stride);

  for (int i = 0; i < PacketSize*20; ++i) {
    if ((i%stride) == 0 && i<stride*PacketSize) {
      VERIFY(isApproxAbs(buffer[i], data1[i/stride], refvalue) && "pscatter");
    } else {
      VERIFY(isApproxAbs(buffer[i], Scalar(0), refvalue) && "pscatter");
    }
  }

  for (int i=0; i<PacketSize*7; ++i) {
    buffer[i] = internal::random<Scalar>()/RealScalar(PacketSize);
  }
  packet = internal::pgather<Scalar, Packet>(buffer, 7);
  internal::pstore(data1, packet);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY(isApproxAbs(data1[i], buffer[i*7], refvalue) && "pgather");
  }
}

void test_packetmath()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( packetmath<float>() );
    CALL_SUBTEST_2( packetmath<double>() );
    CALL_SUBTEST_3( packetmath<int>() );
    CALL_SUBTEST_4( packetmath<std::complex<float> >() );
    CALL_SUBTEST_5( packetmath<std::complex<double> >() );

    CALL_SUBTEST_1( packetmath_notcomplex<float>() );
    CALL_SUBTEST_2( packetmath_notcomplex<double>() );
    CALL_SUBTEST_3( packetmath_notcomplex<int>() );

    CALL_SUBTEST_1( packetmath_real<float>() );
    CALL_SUBTEST_2( packetmath_real<double>() );

    CALL_SUBTEST_4( packetmath_complex<std::complex<float> >() );
    CALL_SUBTEST_5( packetmath_complex<std::complex<double> >() );

    CALL_SUBTEST_1( packetmath_scatter_gather<float>() );
    CALL_SUBTEST_2( packetmath_scatter_gather<double>() );
    CALL_SUBTEST_3( packetmath_scatter_gather<int>() );
    CALL_SUBTEST_4( packetmath_scatter_gather<std::complex<float> >() );
    CALL_SUBTEST_5( packetmath_scatter_gather<std::complex<double> >() );
  }
}
