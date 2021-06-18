/* 
 * File:   gpuBackProjectionKernel.cuh
 * Author: arwillis
 *
 * Created on June 12, 2021, 10:29 AM
 */

#ifndef CPUBACKPROJECTION_HPP
#define CPUBACKPROJECTION_HPP

#include <iostream>
#include <iterator>
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

//template<typename _numTp>
//class Point2 {
//public:
//    _numTp x, y;
//};
//
//template<typename _numTp>
//class Point3 : public Point2<_numTp> {
//public:
//    _numTp z;
//};
//
//template<typename _numTp>
//class Point4 : public Point3<_numTp> {
//public:
//    _numTp w;
//};

template<typename __numTp>
struct simpleMatrix {
public:
    std::vector<int> shape;
    std::vector<__numTp> data;
};

template<typename _numTp>
struct Aperture {
public:
    bool format_GOTCHA;

    // GOTCHA + Sandia Fields
    simpleMatrix<Complex> sampleData;
    //int numPulses;
    //int numRangeSamples;
    simpleMatrix<_numTp> Ant_x;
    simpleMatrix<_numTp> Ant_y;
    simpleMatrix<_numTp> Ant_z;

    // GOTCHA-Only Fields
    simpleMatrix<_numTp> freq;
    simpleMatrix<_numTp> slant_range;
    simpleMatrix<_numTp> theta;
    simpleMatrix<_numTp> phi;

    struct {
        simpleMatrix<_numTp> r_correct;
        simpleMatrix<_numTp> ph_correct;
    } af;

    // Sandia-ONLY Fields
    simpleMatrix<_numTp> ADF;
    simpleMatrix<_numTp> startF;
    simpleMatrix<_numTp> chirpRate;
    simpleMatrix<_numTp> chirpRateDelta;

    template <typename _Tp>
    friend std::ostream& operator<<(std::ostream& output, const Aperture<_Tp> &c);
};

#define streamVec(fieldname, __Tp, sMat, os, nVals) ({\
    os << fieldname << "["; for (auto i:sMat.shape) os << i << 'x'; os << "1] = {"; \
    typename std::vector<__Tp>::const_iterator i; \
    for (i = sMat.data.begin(); \
            i != sMat.data.end() && i != sMat.data.begin() + nVals; ++i) \
            {os << *i << ", ";} }); \
            os << ((sMat.data.size() > nVals) ? " ..." : "") << " }" << std::endl

template <typename _numTp>
inline std::ostream& operator<<(std::ostream& output, const Aperture<_numTp>& c) {
    int NUMVALS = 10;
    if (c.format_GOTCHA) {
        streamVec("sampleData", Complex, c.sampleData, output, NUMVALS);
        streamVec("freq", _numTp, c.freq, output, NUMVALS);
        streamVec("Ant_x", _numTp, c.Ant_x, output, NUMVALS);
        streamVec("Ant_y", _numTp, c.Ant_y, output, NUMVALS);
        streamVec("Ant_z", _numTp, c.Ant_z, output, NUMVALS);
        streamVec("slant_range", _numTp, c.Ant_z, output, NUMVALS);
        streamVec("theta", _numTp, c.Ant_z, output, NUMVALS);
        streamVec("phi", _numTp, c.Ant_z, output, NUMVALS);
        streamVec("af.r_correct", _numTp, c.af.r_correct, output, NUMVALS);
        streamVec("af.ph_correct", _numTp, c.af.ph_correct, output, NUMVALS);
    } else {
        streamVec("sampleData", Complex, c.sampleData, output, NUMVALS);
        streamVec("Ant_x", _numTp, c.Ant_x, output, NUMVALS);
        streamVec("Ant_y", _numTp, c.Ant_y, output, NUMVALS);
        streamVec("Ant_z", _numTp, c.Ant_z, output, NUMVALS);
        streamVec("ADF", _numTp, c.ADF, output, NUMVALS);
        streamVec("StartF", _numTp, c.startF, output, NUMVALS);
        streamVec("ChirpRate", _numTp, c.chirpRate, output, NUMVALS);
        streamVec("ChirpRateDelta", _numTp, c.chirpRateDelta, output, NUMVALS);
    }
    return output;
}

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

