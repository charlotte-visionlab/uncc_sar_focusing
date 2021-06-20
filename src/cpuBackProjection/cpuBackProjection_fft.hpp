/* 
 * File:   cpuBackProjection_fft.hpp
 * Author: arwillis
 *
 * Created on June 19, 2021, 9:51 PM
 */

#ifndef CPUBACKPROJECTION_FFT_HPP
#define CPUBACKPROJECTION_FFT_HPP

#include <fftw3.h>

#include "cpuBackProjection.hpp"

// See cpuBackProjection_fft.cpp for specializations based on numeric type
template<typename __nTp>
void fftw_engine(CArray<__nTp>& x, int DIR) {
    std::cout << "A specialization for this type is not available in FFTW!" << std::endl;
}

template<typename __nTp>
void fftw(CArray<__nTp>& x) {
    fftw_engine(x, FFTW_FORWARD);
}

template<typename __nTp>
void ifftw(CArray<__nTp>& x) {
    fftw_engine(x, FFTW_BACKWARD);
}

template<typename __nTp>
CArray<__nTp> fftshift(CArray<__nTp>& fft) {
    return fft.cshift((fft.size() + 1) / 2);
}

// Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
// Better optimized but less intuitive
// !!! Warning : in some cases this code make result different from not optimized version above (need to fix bug)
// The bug is now fixed @2017/05/30 

template<typename __nTp>
void fft(CArray<__nTp> &x) {
    // DFT
    unsigned int N = x.size(), k = N, n;
    double thetaT = 3.14159265358979323846264338328L / N;
    Complex<__nTp> phiT = Complex<__nTp>(cos(thetaT), -sin(thetaT)), T;
    while (k > 1) {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        T = 1.0L;
        for (unsigned int l = 0; l < k; l++) {
            for (unsigned int a = l; a < N; a += n) {
                unsigned int b = a + k;
                Complex<__nTp> t = x[a] - x[b];
                x[a] += x[b];
                x[b] = t * T;
            }
            T *= phiT;
        }
    }
    // Decimate
    unsigned int m = (unsigned int) log2(N);
    for (unsigned int a = 0; a < N; a++) {
        unsigned int b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        if (b > a) {
            Complex<__nTp> t = x[a];
            x[a] = x[b];
            x[b] = t;
        }
    }
    //// Normalize (This section make it not working correctly)
    //Complex f = 1.0 / sqrt(N);
    //for (unsigned int i = 0; i < N; i++)
    //	x[i] *= f;
}

template<typename __nTp>
void ifft(CArray<__nTp>& x) {
    // conjugate the complex numbers
    x = x.apply(Complex<__nTp>::conj);

    // forward fft
    fft(x);

    // conjugate the complex numbers again
    x = x.apply(Complex<__nTp>::conj);

    // scale the numbers
    x /= x.size();
}

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive

template<typename __nTp>
void fft_alt(CArray<__nTp>& x) {
    const size_t N = x.size();
    if (N <= 1) return;

    // divide
    CArray<__nTp> even = x[std::slice(0, N / 2, 2)];
    CArray<__nTp> odd = x[std::slice(1, N / 2, 2)];

    // conquer
    fft_alt(even);
    fft_alt(odd);

    // combine
    for (size_t k = 0; k < N / 2; ++k) {
        //Complex t = Complex::polar(1.0f, -2.0f * PI * k / N) * odd[k];
        Complex<__nTp> t = Complex<__nTp>::polar(1.0f, -2.0f * PI * k / N) * odd[k];
        x[k ] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

#endif /* CPUBACKPROJECTION_FFT_HPP */

