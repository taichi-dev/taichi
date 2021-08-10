// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Mark Borgerding mark a borgerding net
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include <bench/BenchUtil.h>
#include <complex>
#include <vector>
#include <Eigen/Core>

#include <unsupported/Eigen/FFT>

using namespace Eigen;
using namespace std;


template <typename T>
string nameof();

template <> string nameof<float>() {return "float";}
template <> string nameof<double>() {return "double";}
template <> string nameof<long double>() {return "long double";}

#ifndef TYPE
#define TYPE float
#endif

#ifndef NFFT
#define NFFT 1024
#endif
#ifndef NDATA
#define NDATA 1000000
#endif

using namespace Eigen;

template <typename T>
void bench(int nfft,bool fwd,bool unscaled=false, bool halfspec=false)
{
    typedef typename NumTraits<T>::Real Scalar;
    typedef typename std::complex<Scalar> Complex;
    int nits = NDATA/nfft;
    vector<T> inbuf(nfft);
    vector<Complex > outbuf(nfft);
    FFT< Scalar > fft;

    if (unscaled) {
        fft.SetFlag(fft.Unscaled);
        cout << "unscaled ";
    }
    if (halfspec) {
        fft.SetFlag(fft.HalfSpectrum);
        cout << "halfspec ";
    }


    std::fill(inbuf.begin(),inbuf.end(),0);
    fft.fwd( outbuf , inbuf);

    BenchTimer timer;
    timer.reset();
    for (int k=0;k<8;++k) {
        timer.start();
        if (fwd)
            for(int i = 0; i < nits; i++)
                fft.fwd( outbuf , inbuf);
        else
            for(int i = 0; i < nits; i++)
                fft.inv(inbuf,outbuf);
        timer.stop();
    }

    cout << nameof<Scalar>() << " ";
    double mflops = 5.*nfft*log2((double)nfft) / (1e6 * timer.value() / (double)nits );
    if ( NumTraits<T>::IsComplex ) {
        cout << "complex";
    }else{
        cout << "real   ";
        mflops /= 2;
    }


    if (fwd)
        cout << " fwd";
    else
        cout << " inv";

    cout << " NFFT=" << nfft << "  " << (double(1e-6*nfft*nits)/timer.value()) << " MS/s  " << mflops << "MFLOPS\n";
}

int main(int argc,char ** argv)
{
    bench<complex<float> >(NFFT,true);
    bench<complex<float> >(NFFT,false);
    bench<float>(NFFT,true);
    bench<float>(NFFT,false);
    bench<float>(NFFT,false,true);
    bench<float>(NFFT,false,true,true);

    bench<complex<double> >(NFFT,true);
    bench<complex<double> >(NFFT,false);
    bench<double>(NFFT,true);
    bench<double>(NFFT,false);
    bench<complex<long double> >(NFFT,true);
    bench<complex<long double> >(NFFT,false);
    bench<long double>(NFFT,true);
    bench<long double>(NFFT,false);
    return 0;
}
