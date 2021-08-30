/*
 * Copyright (C) 2021 Andrew R. Willis
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// this declaration needs to be in any C++ compiled target for CPU
#define CUDAFUNCTION

#include "cpuBackProjection_fft.hpp"
//
// Note: FFTW uses completely different functions and libraries
// for double and float implementations. Hence, we specialize the
// template in the header (shown below) to float and double as
// completely different sets of FFTW code blocks. We link against
// both double and float/single precision libraries.
//
//template<typename __nTp>
//void fftw_engine(CArray<__nTp>& x, int DIR) {
//    std::cout << "A specialization for this type is not available in FFTW!" << std::endl;
//}

template<>
void fftw_engine(CArray<float>& x, int DIR) {
    // http://www.fftw.org/fftw3_doc/Complex-numbers.html
    // Structure must be only two numbers in the order real, imag
    // to be binary compatible with the C99 complex type
    //
    // TODO: Mangle function invocations and linking based on float/double selection
    // all calls change to fftw_plan for type double complex numbers
    // and the program must then link against fftw3 not fftw3f
    fftwf_plan p = fftwf_plan_dft_1d(x.size(),
            reinterpret_cast<fftwf_complex*> (&x[0]),
            reinterpret_cast<fftwf_complex*> (&x[0]),
            DIR, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    if (DIR == FFTW_BACKWARD) {
        x /= x.size();
    }
}

template<>
void fftw_engine(CArray<double>& x, int DIR) {
    // http://www.fftw.org/fftw3_doc/Complex-numbers.html
    // Structure must be only two numbers in the order real, imag
    // to be binary compatible with the C99 complex type
    //
    // TODO: Mangle function invocations and linking based on float/double selection
    // all calls change to fftw_plan for type double complex numbers
    // and the program must then link against fftw3 not fftw3f
    fftw_plan p = fftw_plan_dft_1d(x.size(),
            reinterpret_cast<fftw_complex*> (&x[0]),
            reinterpret_cast<fftw_complex*> (&x[0]),
            DIR, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    if (DIR == FFTW_BACKWARD) {
        x /= x.size();
    }
}
