// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_TEST_PART_7)

#ifndef EIGEN_NO_STATIC_ASSERT
#define EIGEN_NO_STATIC_ASSERT // turn static asserts into runtime asserts in order to check them
#endif

// ignore double-promotion diagnostic for clang and gcc, if we check for static assertion anyway:
// TODO do the same for MSVC?
#if defined(__clang__)
#  if (__clang_major__ * 100 + __clang_minor__) >= 308
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#  endif
#elif defined(__GNUC__)
  // TODO is there a minimal GCC version for this? At least g++-4.7 seems to be fine with this.
#  pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#endif



#if defined(EIGEN_TEST_PART_1) || defined(EIGEN_TEST_PART_2) || defined(EIGEN_TEST_PART_3)

#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
#endif

#endif

static bool g_called;
#define EIGEN_SCALAR_BINARY_OP_PLUGIN { g_called |= (!internal::is_same<LhsScalar,RhsScalar>::value); }

#include "main.h"

using namespace std;

#define VERIFY_MIX_SCALAR(XPR,REF) \
  g_called = false; \
  VERIFY_IS_APPROX(XPR,REF); \
  VERIFY( g_called && #XPR" not properly optimized");

template<int SizeAtCompileType>
void raise_assertion(Index size = SizeAtCompileType)
{
  // VERIFY_RAISES_ASSERT(mf+md); // does not even compile
  Matrix<float, SizeAtCompileType, 1> vf; vf.setRandom(size);
  Matrix<double, SizeAtCompileType, 1> vd; vd.setRandom(size);
  VERIFY_RAISES_ASSERT(vf=vd);
  VERIFY_RAISES_ASSERT(vf+=vd);
  VERIFY_RAISES_ASSERT(vf-=vd);
  VERIFY_RAISES_ASSERT(vd=vf);
  VERIFY_RAISES_ASSERT(vd+=vf);
  VERIFY_RAISES_ASSERT(vd-=vf);

  //   vd.asDiagonal() * mf;    // does not even compile
  //   vcd.asDiagonal() * mf;   // does not even compile

#if 0 // we get other compilation errors here than just static asserts
  VERIFY_RAISES_ASSERT(vd.dot(vf));
#endif
}


template<int SizeAtCompileType> void mixingtypes(int size = SizeAtCompileType)
{
  typedef std::complex<float>   CF;
  typedef std::complex<double>  CD;
  typedef Matrix<float, SizeAtCompileType, SizeAtCompileType> Mat_f;
  typedef Matrix<double, SizeAtCompileType, SizeAtCompileType> Mat_d;
  typedef Matrix<std::complex<float>, SizeAtCompileType, SizeAtCompileType> Mat_cf;
  typedef Matrix<std::complex<double>, SizeAtCompileType, SizeAtCompileType> Mat_cd;
  typedef Matrix<float, SizeAtCompileType, 1> Vec_f;
  typedef Matrix<double, SizeAtCompileType, 1> Vec_d;
  typedef Matrix<std::complex<float>, SizeAtCompileType, 1> Vec_cf;
  typedef Matrix<std::complex<double>, SizeAtCompileType, 1> Vec_cd;

  Mat_f mf    = Mat_f::Random(size,size);
  Mat_d md    = mf.template cast<double>();
  //Mat_d rd    = md;
  Mat_cf mcf  = Mat_cf::Random(size,size);
  Mat_cd mcd  = mcf.template cast<complex<double> >();
  Mat_cd rcd = mcd;
  Vec_f vf    = Vec_f::Random(size,1);
  Vec_d vd    = vf.template cast<double>();
  Vec_cf vcf  = Vec_cf::Random(size,1);
  Vec_cd vcd  = vcf.template cast<complex<double> >();
  float           sf  = internal::random<float>();
  double          sd  = internal::random<double>();
  complex<float>  scf = internal::random<complex<float> >();
  complex<double> scd = internal::random<complex<double> >();

  mf+mf;

  float  epsf = std::sqrt(std::numeric_limits<float> ::min EIGEN_EMPTY ());
  double epsd = std::sqrt(std::numeric_limits<double>::min EIGEN_EMPTY ());

  while(std::abs(sf )<epsf) sf  = internal::random<float>();
  while(std::abs(sd )<epsd) sd  = internal::random<double>();
  while(std::abs(scf)<epsf) scf = internal::random<CF>();
  while(std::abs(scd)<epsd) scd = internal::random<CD>();

  // check scalar products
  VERIFY_MIX_SCALAR(vcf * sf , vcf * complex<float>(sf));
  VERIFY_MIX_SCALAR(sd * vcd , complex<double>(sd) * vcd);
  VERIFY_MIX_SCALAR(vf * scf , vf.template cast<complex<float> >() * scf);
  VERIFY_MIX_SCALAR(scd * vd , scd * vd.template cast<complex<double> >());

  VERIFY_MIX_SCALAR(vcf * 2 , vcf * complex<float>(2));
  VERIFY_MIX_SCALAR(vcf * 2.1 , vcf * complex<float>(2.1));
  VERIFY_MIX_SCALAR(2 * vcf, vcf * complex<float>(2));
  VERIFY_MIX_SCALAR(2.1 * vcf , vcf * complex<float>(2.1));

  // check scalar quotients
  VERIFY_MIX_SCALAR(vcf / sf , vcf / complex<float>(sf));
  VERIFY_MIX_SCALAR(vf / scf , vf.template cast<complex<float> >() / scf);
  VERIFY_MIX_SCALAR(vf.array()  / scf, vf.template cast<complex<float> >().array() / scf);
  VERIFY_MIX_SCALAR(scd / vd.array() , scd / vd.template cast<complex<double> >().array());

  // check scalar increment
  VERIFY_MIX_SCALAR(vcf.array() + sf , vcf.array() + complex<float>(sf));
  VERIFY_MIX_SCALAR(sd  + vcd.array(), complex<double>(sd) + vcd.array());
  VERIFY_MIX_SCALAR(vf.array()  + scf, vf.template cast<complex<float> >().array() + scf);
  VERIFY_MIX_SCALAR(scd + vd.array() , scd + vd.template cast<complex<double> >().array());

  // check scalar subtractions
  VERIFY_MIX_SCALAR(vcf.array() - sf , vcf.array() - complex<float>(sf));
  VERIFY_MIX_SCALAR(sd  - vcd.array(), complex<double>(sd) - vcd.array());
  VERIFY_MIX_SCALAR(vf.array()  - scf, vf.template cast<complex<float> >().array() - scf);
  VERIFY_MIX_SCALAR(scd - vd.array() , scd - vd.template cast<complex<double> >().array());

  // check scalar powers
  VERIFY_MIX_SCALAR( pow(vcf.array(), sf),        Eigen::pow(vcf.array(), complex<float>(sf)) );
  VERIFY_MIX_SCALAR( vcf.array().pow(sf) ,        Eigen::pow(vcf.array(), complex<float>(sf)) );
  VERIFY_MIX_SCALAR( pow(sd, vcd.array()),        Eigen::pow(complex<double>(sd), vcd.array()) );
  VERIFY_MIX_SCALAR( Eigen::pow(vf.array(), scf), Eigen::pow(vf.template cast<complex<float> >().array(), scf) );
  VERIFY_MIX_SCALAR( vf.array().pow(scf) ,        Eigen::pow(vf.template cast<complex<float> >().array(), scf) );
  VERIFY_MIX_SCALAR( Eigen::pow(scd, vd.array()), Eigen::pow(scd, vd.template cast<complex<double> >().array()) );

  // check dot product
  vf.dot(vf);
  VERIFY_IS_APPROX(vcf.dot(vf), vcf.dot(vf.template cast<complex<float> >()));

  // check diagonal product
  VERIFY_IS_APPROX(vf.asDiagonal() * mcf, vf.template cast<complex<float> >().asDiagonal() * mcf);
  VERIFY_IS_APPROX(vcd.asDiagonal() * md, vcd.asDiagonal() * md.template cast<complex<double> >());
  VERIFY_IS_APPROX(mcf * vf.asDiagonal(), mcf * vf.template cast<complex<float> >().asDiagonal());
  VERIFY_IS_APPROX(md * vcd.asDiagonal(), md.template cast<complex<double> >() * vcd.asDiagonal());

  // check inner product
  VERIFY_IS_APPROX((vf.transpose() * vcf).value(), (vf.template cast<complex<float> >().transpose() * vcf).value());

  // check outer product
  VERIFY_IS_APPROX((vf * vcf.transpose()).eval(), (vf.template cast<complex<float> >() * vcf.transpose()).eval());

  // coeff wise product

  VERIFY_IS_APPROX((vf * vcf.transpose()).eval(), (vf.template cast<complex<float> >() * vcf.transpose()).eval());

  Mat_cd mcd2 = mcd;
  VERIFY_IS_APPROX(mcd.array() *= md.array(), mcd2.array() *= md.array().template cast<std::complex<double> >());
  
  // check matrix-matrix products
  VERIFY_IS_APPROX(sd*md*mcd, (sd*md).template cast<CD>().eval()*mcd);
  VERIFY_IS_APPROX(sd*mcd*md, sd*mcd*md.template cast<CD>());
  VERIFY_IS_APPROX(scd*md*mcd, scd*md.template cast<CD>().eval()*mcd);
  VERIFY_IS_APPROX(scd*mcd*md, scd*mcd*md.template cast<CD>());

  VERIFY_IS_APPROX(sf*mf*mcf, sf*mf.template cast<CF>()*mcf);
  VERIFY_IS_APPROX(sf*mcf*mf, sf*mcf*mf.template cast<CF>());
  VERIFY_IS_APPROX(scf*mf*mcf, scf*mf.template cast<CF>()*mcf);
  VERIFY_IS_APPROX(scf*mcf*mf, scf*mcf*mf.template cast<CF>());

  VERIFY_IS_APPROX(sd*md.adjoint()*mcd, (sd*md).template cast<CD>().eval().adjoint()*mcd);
  VERIFY_IS_APPROX(sd*mcd.adjoint()*md, sd*mcd.adjoint()*md.template cast<CD>());
  VERIFY_IS_APPROX(sd*md.adjoint()*mcd.adjoint(), (sd*md).template cast<CD>().eval().adjoint()*mcd.adjoint());
  VERIFY_IS_APPROX(sd*mcd.adjoint()*md.adjoint(), sd*mcd.adjoint()*md.template cast<CD>().adjoint());
  VERIFY_IS_APPROX(sd*md*mcd.adjoint(), (sd*md).template cast<CD>().eval()*mcd.adjoint());
  VERIFY_IS_APPROX(sd*mcd*md.adjoint(), sd*mcd*md.template cast<CD>().adjoint());

  VERIFY_IS_APPROX(sf*mf.adjoint()*mcf, (sf*mf).template cast<CF>().eval().adjoint()*mcf);
  VERIFY_IS_APPROX(sf*mcf.adjoint()*mf, sf*mcf.adjoint()*mf.template cast<CF>());
  VERIFY_IS_APPROX(sf*mf.adjoint()*mcf.adjoint(), (sf*mf).template cast<CF>().eval().adjoint()*mcf.adjoint());
  VERIFY_IS_APPROX(sf*mcf.adjoint()*mf.adjoint(), sf*mcf.adjoint()*mf.template cast<CF>().adjoint());
  VERIFY_IS_APPROX(sf*mf*mcf.adjoint(), (sf*mf).template cast<CF>().eval()*mcf.adjoint());
  VERIFY_IS_APPROX(sf*mcf*mf.adjoint(), sf*mcf*mf.template cast<CF>().adjoint());

  VERIFY_IS_APPROX(sf*mf*vcf, (sf*mf).template cast<CF>().eval()*vcf);
  VERIFY_IS_APPROX(scf*mf*vcf,(scf*mf.template cast<CF>()).eval()*vcf);
  VERIFY_IS_APPROX(sf*mcf*vf, sf*mcf*vf.template cast<CF>());
  VERIFY_IS_APPROX(scf*mcf*vf,scf*mcf*vf.template cast<CF>());

  VERIFY_IS_APPROX(sf*vcf.adjoint()*mf,  sf*vcf.adjoint()*mf.template cast<CF>().eval());
  VERIFY_IS_APPROX(scf*vcf.adjoint()*mf, scf*vcf.adjoint()*mf.template cast<CF>().eval());
  VERIFY_IS_APPROX(sf*vf.adjoint()*mcf,  sf*vf.adjoint().template cast<CF>().eval()*mcf);
  VERIFY_IS_APPROX(scf*vf.adjoint()*mcf, scf*vf.adjoint().template cast<CF>().eval()*mcf);

  VERIFY_IS_APPROX(sd*md*vcd, (sd*md).template cast<CD>().eval()*vcd);
  VERIFY_IS_APPROX(scd*md*vcd,(scd*md.template cast<CD>()).eval()*vcd);
  VERIFY_IS_APPROX(sd*mcd*vd, sd*mcd*vd.template cast<CD>().eval());
  VERIFY_IS_APPROX(scd*mcd*vd,scd*mcd*vd.template cast<CD>().eval());

  VERIFY_IS_APPROX(sd*vcd.adjoint()*md,  sd*vcd.adjoint()*md.template cast<CD>().eval());
  VERIFY_IS_APPROX(scd*vcd.adjoint()*md, scd*vcd.adjoint()*md.template cast<CD>().eval());
  VERIFY_IS_APPROX(sd*vd.adjoint()*mcd,  sd*vd.adjoint().template cast<CD>().eval()*mcd);
  VERIFY_IS_APPROX(scd*vd.adjoint()*mcd, scd*vd.adjoint().template cast<CD>().eval()*mcd);

  VERIFY_IS_APPROX( sd*vcd.adjoint()*md.template triangularView<Upper>(),  sd*vcd.adjoint()*md.template cast<CD>().eval().template triangularView<Upper>());
  VERIFY_IS_APPROX(scd*vcd.adjoint()*md.template triangularView<Lower>(), scd*vcd.adjoint()*md.template cast<CD>().eval().template triangularView<Lower>());
  VERIFY_IS_APPROX( sd*vcd.adjoint()*md.transpose().template triangularView<Upper>(),  sd*vcd.adjoint()*md.transpose().template cast<CD>().eval().template triangularView<Upper>());
  VERIFY_IS_APPROX(scd*vcd.adjoint()*md.transpose().template triangularView<Lower>(), scd*vcd.adjoint()*md.transpose().template cast<CD>().eval().template triangularView<Lower>());
  VERIFY_IS_APPROX( sd*vd.adjoint()*mcd.template triangularView<Lower>(),  sd*vd.adjoint().template cast<CD>().eval()*mcd.template triangularView<Lower>());
  VERIFY_IS_APPROX(scd*vd.adjoint()*mcd.template triangularView<Upper>(), scd*vd.adjoint().template cast<CD>().eval()*mcd.template triangularView<Upper>());
  VERIFY_IS_APPROX( sd*vd.adjoint()*mcd.transpose().template triangularView<Lower>(),  sd*vd.adjoint().template cast<CD>().eval()*mcd.transpose().template triangularView<Lower>());
  VERIFY_IS_APPROX(scd*vd.adjoint()*mcd.transpose().template triangularView<Upper>(), scd*vd.adjoint().template cast<CD>().eval()*mcd.transpose().template triangularView<Upper>());

  // Not supported yet: trmm
//   VERIFY_IS_APPROX(sd*mcd*md.template triangularView<Lower>(),  sd*mcd*md.template cast<CD>().eval().template triangularView<Lower>());
//   VERIFY_IS_APPROX(scd*mcd*md.template triangularView<Upper>(), scd*mcd*md.template cast<CD>().eval().template triangularView<Upper>());
//   VERIFY_IS_APPROX(sd*md*mcd.template triangularView<Lower>(),  sd*md.template cast<CD>().eval()*mcd.template triangularView<Lower>());
//   VERIFY_IS_APPROX(scd*md*mcd.template triangularView<Upper>(), scd*md.template cast<CD>().eval()*mcd.template triangularView<Upper>());

  // Not supported yet: symv
//   VERIFY_IS_APPROX(sd*vcd.adjoint()*md.template selfadjointView<Upper>(),  sd*vcd.adjoint()*md.template cast<CD>().eval().template selfadjointView<Upper>());
//   VERIFY_IS_APPROX(scd*vcd.adjoint()*md.template selfadjointView<Lower>(), scd*vcd.adjoint()*md.template cast<CD>().eval().template selfadjointView<Lower>());
//   VERIFY_IS_APPROX(sd*vd.adjoint()*mcd.template selfadjointView<Lower>(),  sd*vd.adjoint().template cast<CD>().eval()*mcd.template selfadjointView<Lower>());
//   VERIFY_IS_APPROX(scd*vd.adjoint()*mcd.template selfadjointView<Upper>(), scd*vd.adjoint().template cast<CD>().eval()*mcd.template selfadjointView<Upper>());

  // Not supported yet: symm
//   VERIFY_IS_APPROX(sd*vcd.adjoint()*md.template selfadjointView<Upper>(),  sd*vcd.adjoint()*md.template cast<CD>().eval().template selfadjointView<Upper>());
//   VERIFY_IS_APPROX(scd*vcd.adjoint()*md.template selfadjointView<Upper>(), scd*vcd.adjoint()*md.template cast<CD>().eval().template selfadjointView<Upper>());
//   VERIFY_IS_APPROX(sd*vd.adjoint()*mcd.template selfadjointView<Upper>(),  sd*vd.adjoint().template cast<CD>().eval()*mcd.template selfadjointView<Upper>());
//   VERIFY_IS_APPROX(scd*vd.adjoint()*mcd.template selfadjointView<Upper>(), scd*vd.adjoint().template cast<CD>().eval()*mcd.template selfadjointView<Upper>());

  rcd.setZero();
  VERIFY_IS_APPROX(Mat_cd(rcd.template triangularView<Upper>() = sd * mcd * md),
                   Mat_cd((sd * mcd * md.template cast<CD>().eval()).template triangularView<Upper>()));
  VERIFY_IS_APPROX(Mat_cd(rcd.template triangularView<Upper>() = sd * md * mcd),
                   Mat_cd((sd * md.template cast<CD>().eval() * mcd).template triangularView<Upper>()));
  VERIFY_IS_APPROX(Mat_cd(rcd.template triangularView<Upper>() = scd * mcd * md),
                   Mat_cd((scd * mcd * md.template cast<CD>().eval()).template triangularView<Upper>()));
  VERIFY_IS_APPROX(Mat_cd(rcd.template triangularView<Upper>() = scd * md * mcd),
                   Mat_cd((scd * md.template cast<CD>().eval() * mcd).template triangularView<Upper>()));


  VERIFY_IS_APPROX( md.array()  * mcd.array(), md.template cast<CD>().eval().array() * mcd.array() );
  VERIFY_IS_APPROX( mcd.array() * md.array(),  mcd.array() * md.template cast<CD>().eval().array() );

  VERIFY_IS_APPROX( md.array()  + mcd.array(), md.template cast<CD>().eval().array() + mcd.array() );
  VERIFY_IS_APPROX( mcd.array() + md.array(),  mcd.array() + md.template cast<CD>().eval().array() );

  VERIFY_IS_APPROX( md.array()  - mcd.array(), md.template cast<CD>().eval().array() - mcd.array() );
  VERIFY_IS_APPROX( mcd.array() - md.array(),  mcd.array() - md.template cast<CD>().eval().array() );

  if(mcd.array().abs().minCoeff()>epsd)
  {
    VERIFY_IS_APPROX( md.array() / mcd.array(), md.template cast<CD>().eval().array() / mcd.array() );
  }
  if(md.array().abs().minCoeff()>epsd)
  {
    VERIFY_IS_APPROX( mcd.array() / md.array(), mcd.array() / md.template cast<CD>().eval().array() );
  }

  if(md.array().abs().minCoeff()>epsd || mcd.array().abs().minCoeff()>epsd)
  {
    VERIFY_IS_APPROX( md.array().pow(mcd.array()), md.template cast<CD>().eval().array().pow(mcd.array()) );
    VERIFY_IS_APPROX( mcd.array().pow(md.array()),  mcd.array().pow(md.template cast<CD>().eval().array()) );

    VERIFY_IS_APPROX( pow(md.array(),mcd.array()), md.template cast<CD>().eval().array().pow(mcd.array()) );
    VERIFY_IS_APPROX( pow(mcd.array(),md.array()),  mcd.array().pow(md.template cast<CD>().eval().array()) );
  }

  rcd = mcd;
  VERIFY_IS_APPROX( rcd = md, md.template cast<CD>().eval() );
  rcd = mcd;
  VERIFY_IS_APPROX( rcd += md, mcd + md.template cast<CD>().eval() );
  rcd = mcd;
  VERIFY_IS_APPROX( rcd -= md, mcd - md.template cast<CD>().eval() );
  rcd = mcd;
  VERIFY_IS_APPROX( rcd.array() *= md.array(), mcd.array() * md.template cast<CD>().eval().array() );
  rcd = mcd;
  if(md.array().abs().minCoeff()>epsd)
  {
    VERIFY_IS_APPROX( rcd.array() /= md.array(), mcd.array() / md.template cast<CD>().eval().array() );
  }

  rcd = mcd;
  VERIFY_IS_APPROX( rcd.noalias() += md + mcd*md, mcd + (md.template cast<CD>().eval()) + mcd*(md.template cast<CD>().eval()));

  VERIFY_IS_APPROX( rcd.noalias()  = md*md,       ((md*md).eval().template cast<CD>()) );
  rcd = mcd;
  VERIFY_IS_APPROX( rcd.noalias() += md*md, mcd + ((md*md).eval().template cast<CD>()) );
  rcd = mcd;
  VERIFY_IS_APPROX( rcd.noalias() -= md*md, mcd - ((md*md).eval().template cast<CD>()) );

  VERIFY_IS_APPROX( rcd.noalias()  = mcd + md*md,       mcd + ((md*md).eval().template cast<CD>()) );
  rcd = mcd;
  VERIFY_IS_APPROX( rcd.noalias() += mcd + md*md, mcd + mcd + ((md*md).eval().template cast<CD>()) );
  rcd = mcd;
  VERIFY_IS_APPROX( rcd.noalias() -= mcd + md*md,           - ((md*md).eval().template cast<CD>()) );
}

void test_mixingtypes()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(mixingtypes<3>());
    CALL_SUBTEST_2(mixingtypes<4>());
    CALL_SUBTEST_3(mixingtypes<Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)));

    CALL_SUBTEST_4(mixingtypes<3>());
    CALL_SUBTEST_5(mixingtypes<4>());
    CALL_SUBTEST_6(mixingtypes<Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)));
    CALL_SUBTEST_7(raise_assertion<Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)));
  }
  CALL_SUBTEST_7(raise_assertion<0>());
  CALL_SUBTEST_7(raise_assertion<3>());
  CALL_SUBTEST_7(raise_assertion<4>());
  CALL_SUBTEST_7(raise_assertion<Dynamic>(0));
}
