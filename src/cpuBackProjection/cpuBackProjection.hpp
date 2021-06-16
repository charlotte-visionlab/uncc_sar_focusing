/* 
 * File:   gpuBackProjectionKernel.cuh
 * Author: arwillis
 *
 * Created on June 12, 2021, 10:29 AM
 */

#ifndef CPUBACKPROJECTION_HPP
#define CPUBACKPROJECTION_HPP

#include <iostream>
#include <cmath>
#include <valarray>
#include <vector>

#include "mxComplexSingleClass.hpp"

//class mxComplexSingleClass;

//#ifndef NO_MATLAB
//#include <mex.h>    // Matlab library includes
//#include <matrix.h> // Matlab mxComplexSingle struct
//
//typedef mxComplexSingleClass Complex;
//#define polarToComplex mxComplexSingleClass::polar
//#define conjugateComplex mxComplexSingleClass::conj
//
//#else
//typedef float mxSingle;
//
//typedef struct {
//    mxSingle real, imag;
//} mxComplexSingle;
//
////#include <complex>
////typedef std::complex<float> Complex;
////#define polarToComplex std::polar
////#define conjugateComplex std::conj
//
//typedef mxComplexSingleClass Complex;
//#define polarToComplex mxComplexSingleClass::polar
//#define conjugateComplex mxComplexSingleClass::conj
//
//#endif

typedef std::valarray<Complex> CArray;
typedef std::vector<Complex> CVector;

class float2 {
public:
    float x, y;
};

class float4 : public float2 {
public:
    float z, w;
};

#define REAL(vec) (vec.x)
#define IMAG(vec) (vec.y)

#define CAREFUL_AMINUSB_SQ(x,y) __fmul_rn(__fadd_rn((x), -1.0f*(y)), __fadd_rn((x), -1.0f*(y)))

#define BLOCKWIDTH    16
#define BLOCKHEIGHT   16

#define MAKERADIUS(xpixel,ypixel, xa,ya,za) sqrtf(CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + __fmul_rn(za, za))

//#define CLIGHT 299792458.0 /* c: speed of light, m/s */
#define CLIGHT 299792458.0f /* c: speed of light, m/s */
#define PI 3.14159265359f   /* pi, accurate to 6th place in single precision */

/***
 * Function Prototypes
 * ***/

void run_bp(const CArray& phd, float* xObs, float* yObs, float* zObs, float* r,
        int Npulses, int Nrangebins, int Nx_pix, int Ny_pix, int Nfft,
        int blockwidth, int blockheight,
        int deviceId, CArray& output_image,
        int start_output_index, int num_output_rows,
        float c__4_delta_freq, float pi_4_f0__clight, float* minF, float* deltaF,
        float x0, float y0, float Wx, float Wy,
        float min_eff_idx, float total_proj_length);

void computeDifferentialRangeAndPhaseCorrections(const float* xObs, const float* yObs, const float* zObs,
        const float* range_to_phasectr, const int pulseIndex, const float* minF,
        const int Npulses, const int Nrangebins, const int Nx_pix, const int Ny_pix, const int Nfft,
        const float x0, const float y0, const float Wx, const float Wy,
        const float* r_vec, const CArray& rangeCompressed, 
        const float min_Rvec, const float max_Rvec, const float maxWr,
        CArray& output_image);

//typedef struct {
//    float * real;
//    float * imag;
//} complex_split;
//
///* To work seamlessly with Hartley's codebase */
//typedef complex_split bp_complex_split;
//float2* format_complex_to_columns(bp_complex_split a, int width_orig,
//        int height_orig);
//
//float2* format_complex(bp_complex_split a, int size);
//
//float4* format_x_y_z_r(float * x, float * y, float * z, float * r, int size);
//
//float2 expjf(float in);
//
//float2 expjf_div_2(float in);

#endif /* CPUBACKPROJECTION_HPP */

