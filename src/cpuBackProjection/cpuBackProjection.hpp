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
#include <string>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "mxComplexSingleClass.hpp"

typedef std::valarray<Complex> CArray;
typedef std::vector<Complex> CVector;

class float2 {
public:
    float x, y;
};

class float3 : public float2 {
public:
    float z;
};

class float4 : public float3 {
public:
    float w;
};

struct PulseData {
public:
    std::unordered_map<std::string, Complex *> sampleData;
    
};

struct Aperture {
public:
    std::vector<PulseData> pulseData;
};

//#define CLIGHT 299792458.0 /* c: speed of light, m/s */
#define CLIGHT 299792458.0f /* c: speed of light, m/s */
#define PI 3.14159265359f   /* pi, accurate to 6th place in single precision */

/***
 * Function Prototypes
 * ***/

void run_bp(const CArray& phd, float* xObs, float* yObs, float* zObs, float* r,
        int Npulses, int Nrangebins, int Nx_pix, int Ny_pix, int Nfft,
        CArray& output_image, float* minF, float* deltaF,
        float x0, float y0, float Wx, float Wy,
        float min_eff_idx, float total_proj_length);

void computeDifferentialRangeAndPhaseCorrections(const float* xObs, const float* yObs, const float* zObs,
        const float* range_to_phasectr, const int pulseIndex, const float* minF,
        const int Npulses, const int Nrangebins, const int Nx_pix, const int Ny_pix, const int Nfft,
        const float x0, const float y0, const float Wx, const float Wy,
        const float* r_vec, const CArray& rangeCompressed, 
        const float min_Rvec, const float max_Rvec, const float maxWr,
        CArray& output_image);


void fft(CArray& x);
void ifft(CArray& x);
void fftw(CArray& x);
void ifftw(CArray& x);

CArray fftshift(CArray& fft);


#endif /* CPUBACKPROJECTION_HPP */

