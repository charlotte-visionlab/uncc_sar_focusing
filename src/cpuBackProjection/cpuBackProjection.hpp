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
#include <numeric>
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

template<typename _numTp>
class SAR_ImageFormationParameters {
public:
    int pol; // What polarization to image (HH,HV,VH,VV)

    // Define image parameters here
    int N_fft; // Number of samples in FFT
    int N_xpix; // Number of samples in x direction
    int N_ypix; // Number of samples in y direction
    _numTp x0; // Center of image scene in x direction (m) relative to target swath phase center
    _numTp y0; // Center of image scene in y direction (m) relative to target swath phase center
    _numTp dyn_range; // dB range [0,...,-dyn_range] as the dynamic range to display/map to 0-255 grayscale intensity
};

template<typename __numTp>
struct simpleMatrix {
public:
    std::vector<int> shape;
    std::vector<__numTp> data;

    int numValues(int polarityIdx = -1) {
        return (shape.size() == 0) ? 0 :
                std::accumulate(begin(shape), end(shape), 1, std::multiplies<int>()) / (polarityIdx == -1 ? 1 : shape[polarityIdx]);
    }

    bool isEmpty() {
        return shape.size() == 0;
    }

    template <typename _Tp>
    friend std::ostream& operator<<(std::ostream& output, const simpleMatrix<__numTp> &c);

};

template <typename __numTp>
inline std::ostream& operator<<(std::ostream& output, const simpleMatrix<__numTp>& sMat) {
    int NUMVALS = 10;
    std::string dimsStr;
    if (!sMat.shape.empty()) {
        dimsStr = std::accumulate(sMat.shape.begin() + 1, sMat.shape.end(),
                std::to_string(sMat.shape[0]), [](const std::string & a, int b) {
                    return a + ',' + std::to_string(b);
                });
    }
    output << "[" << dimsStr << "] = {"; \
    typename std::vector<__numTp>::const_iterator i;
    for (i = sMat.data.begin(); i != sMat.data.end() && i != sMat.data.begin() + NUMVALS; ++i) {
        output << *i << ", ";
    }
    output << ((sMat.data.size() > NUMVALS) ? " ..." : "") << " }";
    return output;
}

template<typename _numTp>
class SAR_Aperture {
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

    // Fields set automatically by program computations or manually via user input arguments
    // 1 - HH, 2 -HV, 3 - VH, 4 - VV
    int polarity_channel;
    // array index to use for polarity data when indexing multi-dimensional arrays
    // -1 = there is only one polarity in the SAR data file
    int polarity_dimension;

    // Fields set automatically
    int numRangeSamples;
    int numAzimuthSamples;
    int numPolarities;

    SAR_Aperture() : polarity_channel(1), polarity_dimension(-1) {
    };

    virtual ~SAR_Aperture() {
    };

    template <typename _Tp>
    friend std::ostream& operator<<(std::ostream& output, const SAR_Aperture<_Tp> &c);
};

template <typename _numTp>
inline std::ostream& operator<<(std::ostream& output, const SAR_Aperture<_numTp>& c) {
    int NUMVALS = 10;
    if (c.format_GOTCHA) {
        output << c.sampleData;
        output << "sampleData" << c.sampleData << std::endl;
        output << "freq" << c.freq << std::endl;
        output << "Ant_x" << c.Ant_x << std::endl;
        output << "Ant_y" << c.Ant_y << std::endl;
        output << "Ant_z" << c.Ant_z << std::endl;
        output << "slant_range" << c.slant_range << std::endl;
        output << "theta" << c.theta << std::endl;
        output << "phi" << c.phi << std::endl;
        output << "af.r_correct" << c.af.r_correct << std::endl;
        output << "af.ph_correct" << c.af.ph_correct << std::endl;
    } else {
        output << "sampleData" << c.sampleData << std::endl;
        output << "Ant_x" << c.Ant_x << std::endl;
        output << "Ant_y" << c.Ant_y << std::endl;
        output << "Ant_z" << c.Ant_z << std::endl;
        output << "ADF" << c.ADF << std::endl;
        output << "StartF" << c.startF << std::endl;
        output << "ChirpRate" << c.chirpRate << std::endl;
        output << "ChirpRateDelta" << c.chirpRateDelta << std::endl;
        output << "freq" << c.freq << std::endl;
        output << "slant_range" << c.slant_range << std::endl;
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

