/* 
 * File:   gpuBackProjectionKernel.cuh
 * Author: arwillis
 *
 * Created on June 12, 2021, 10:29 AM
 */

#ifndef CPUBACKPROJECTION_HPP
#define CPUBACKPROJECTION_HPP

#include <iomanip>
#include <iostream>
#include <iterator>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "mxComplexSingleClass.hpp"


#define CLIGHT 299792458.0 /* c: speed of light, m/s */
//#define CLIGHT 299792458.0f /* c: speed of light, m/s */
#define PI 3.14159265359f   /* pi, accurate to 6th place in single precision */

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

    int numValues(int polarityIdx = -1) {
        if (polarityIdx > shape.size()) {
            polarityIdx = -1;
        }
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
    simpleMatrix<_numTp> bandwidth;
    simpleMatrix<_numTp> deltaF;

    _numTp mean_startF;
    _numTp mean_deltaF;
    _numTp mean_bandwidth;

    simpleMatrix<_numTp> Ant_Az;
    simpleMatrix<_numTp> Ant_deltaAz;
    simpleMatrix<_numTp> Ant_El;
    simpleMatrix<_numTp> Ant_deltaEl;
    _numTp mean_Ant_El;
    _numTp mean_deltaAz;
    _numTp mean_deltaEl;
    _numTp mean_Ant_deltaAz;
    _numTp mean_Ant_deltaEl;
    _numTp Ant_totalAz;
    _numTp Ant_totalEl;

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
    output << "sampleData" << c.sampleData << std::endl;
    output << "Ant_x" << c.Ant_x << std::endl;
    output << "Ant_y" << c.Ant_y << std::endl;
    output << "Ant_z" << c.Ant_z << std::endl;
    output << "Ant_Az" << c.Ant_Az << std::endl;
    output << "Ant_El" << c.Ant_El << std::endl;
    output << "Ant_deltaAz" << c.Ant_deltaAz << std::endl;
    output << "Ant_totalAz = " << c.Ant_totalAz << std::endl;
    output << "Ant_totalEl = " << c.Ant_totalEl << std::endl;
    output << "freq" << c.freq << std::endl;
    output << "StartF" << c.startF << std::endl;
    output << "deltaF" << c.deltaF << std::endl;
    output << "bandwidth" << c.bandwidth << std::endl;
    output << "slant_range" << c.slant_range << std::endl;
    if (c.format_GOTCHA) {
        output << "theta" << c.theta << std::endl;
        output << "phi" << c.phi << std::endl;
        output << "af.r_correct" << c.af.r_correct << std::endl;
        output << "af.ph_correct" << c.af.ph_correct << std::endl;
    } else {
        output << "ADF" << c.ADF << std::endl;
        output << "StartF" << c.startF << std::endl;
        output << "ChirpRate" << c.chirpRate << std::endl;
        output << "ChirpRateDelta" << c.chirpRateDelta << std::endl;
    }
    return output;
}

template<typename _numTp>
class SAR_ImageFormationParameters {
public:
    int pol; // What polarization to image (HH,HV,VH,VV)
    std::string output_filename; // Filename to write output image
    // Define image parameters here
    int N_fft; // Number of samples in FFT
    int N_x_pix; // Number of samples in x direction
    int N_y_pix; // Number of samples in y direction
    _numTp x0_m; // Center of image scene in x direction (m) relative to target swath phase center
    _numTp y0_m; // Center of image scene in y direction (m) relative to target swath phase center
    _numTp Wx_m; // Extent of the focused image scene about (x0,y0) in cross-range/x direction (m)
    _numTp Wy_m; // Extent of the focused image scene about (x0,y0) in down-range/y direction (m)
    _numTp max_Wx_m; // Maximum extent of image scene in cross-range/x direction (m) about (x0,y0) = (0,0)
    _numTp max_Wy_m; // Maximum extent of image scene in down-range/y direction (m) about (x0,y0) = (0,0)
    _numTp dyn_range_dB; // dB range [0,...,-dyn_range] as the dynamic range to display/map to 0-255 grayscale intensity
    _numTp slant_rangeResolution; // Slant range resolution in the down-range/x direction (m)
    _numTp ground_rangeResolution; // Ground range resolution in the down-range/x direction (m)
    _numTp azimuthResolution; // Resolution in the cross-range/x direction (m)

    SAR_ImageFormationParameters() : N_fft(512), N_x_pix(512), N_y_pix(512), x0_m(0), y0_m(0), dyn_range_dB(70) {
    };

    template <typename __myTp, typename __argTp>
    static SAR_ImageFormationParameters create(SAR_Aperture<__argTp> aperture) {
        // call the constructor
        SAR_ImageFormationParameters<__myTp> image_params;

        image_params.N_fft = aperture.numRangeSamples;
        image_params.N_x_pix = aperture.numAzimuthSamples;
        image_params.N_y_pix = image_params.N_fft;
        // focus image on target phase center
        // Redundant with constructor
        //image_params.x0_m = 0;
        //image_params.y0_m = 0;
        // Determine the maximum scene size of the image (m)
        // max down-range/fast-time/y-axis extent of image (m)
        image_params.max_Wy_m = CLIGHT / (2.0 * aperture.mean_deltaF);
        // max cross-range/fast-time/x-axis extent of image (m)
        image_params.max_Wx_m = CLIGHT / (2.0 * std::abs(aperture.mean_Ant_deltaAz) * aperture.mean_startF);

        // default view is 100% of the maximum possible view
        image_params.Wx_m = 1.00 * image_params.max_Wx_m;
        image_params.Wy_m = 1.00 * image_params.max_Wy_m;
        // make reconstructed image equal size in (x,y) dimensions
        image_params.N_x_pix = (int) ((float) image_params.Wx_m * image_params.N_y_pix) / image_params.Wy_m;
        // Determine the resolution of the image (m)
        image_params.slant_rangeResolution = CLIGHT / (2.0 * aperture.mean_bandwidth);
        image_params.ground_rangeResolution = image_params.slant_rangeResolution / std::sin(aperture.mean_Ant_El);
        image_params.azimuthResolution = CLIGHT / (2.0 * aperture.Ant_totalAz * aperture.mean_startF);

        return image_params;
    }

    virtual ~SAR_ImageFormationParameters() {
    };

    template <typename _Tp>
    friend std::ostream& operator<<(std::ostream& output, const SAR_ImageFormationParameters<_Tp> &c);
};

template <typename _numTp>
inline std::ostream& operator<<(std::ostream& output, const SAR_ImageFormationParameters<_numTp>& c) {
    output << "Nfft = {" << c.N_fft << "}" << std::endl;
    output << "N_x_pix = {" << c.N_x_pix << "}" << std::endl;
    output << "N_y_pix = {" << c.N_y_pix << "}" << std::endl;
    output << "x0_m = {" << c.x0_m << "}" << std::endl;
    output << "y0_m = {" << c.y0_m << "}" << std::endl;
    output << "max_Wx_m = {" << c.max_Wx_m << "}" << std::endl;
    output << "max_Wy_m = {" << c.max_Wy_m << "}" << std::endl;
    output << "Wx_m = {" << c.Wx_m << "}" << std::endl;
    output << "Wy_m = {" << c.Wy_m << "}" << std::endl;
    output << "deltaR_m (slant range resolution)= {" << c.slant_rangeResolution << "}" << std::endl;
    output << "deltaX_m (ground range resolution)= {" << c.ground_rangeResolution << "}" << std::endl;
    output << "deltaY_m (cross-range/x-axis resolution) = {" << c.azimuthResolution << "}" << std::endl;
}

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

template <typename __nTp>
Complex interp1(const __nTp* xSampleLocations, const int nSamples, const CArray& sampleValues, const float xInterpLocation, const __nTp xIndex) {
    Complex iVal(0, 0);
    int rightIdx = std::floor(xIndex) - 1;
    while (++rightIdx < nSamples && xSampleLocations[rightIdx] <= xInterpLocation);
    if (rightIdx == nSamples || rightIdx == 0) {
        std::cout << "Error::Invalid interpolation range." << std::endl;
        return iVal;
    }
    //if (rightIdx < (int) std::floor(xIndex)) {
    //    std::cout << "Error incorrect predicted location for dR. rightIdx = " << rightIdx << " dR_Idx= " << std::ceil(xIndex) << std::endl;
    //}
    __nTp alpha = (xInterpLocation - xSampleLocations[rightIdx - 1]) / (xSampleLocations[rightIdx] - xSampleLocations[rightIdx - 1]);
    iVal = alpha * sampleValues[rightIdx] + (1.0f - alpha) * sampleValues[rightIdx - 1];
    return iVal;
}

// idx should be integer    
#define RANGE_INDEX_TO_RANGE_VALUE(idx, maxWr, N) ((float) idx / N - 0.5f) * maxWr
// val should be float
#define RANGE_VALUE_TO_RANGE_INDEX(val, maxWr, N) (val / maxWr + 0.5f) * N

template <typename __nTpData, typename __nTpParams>
void computeDifferentialRangeAndPhaseCorrections(int pulseIndex,
        const SAR_Aperture<__nTpData>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        const CArray& rangeCompressed,
        const __nTpData* r_vec, const __nTpData min_Rvec, const __nTpData max_Rvec,
        CArray& output_image) {

    float4 target;
    target.x = SARImgParams.x0_m - (SARImgParams.Wx_m / 2);
    target.z = 0;
    float delta_x = SARImgParams.Wx_m / (SARImgParams.N_x_pix - 1);
    float delta_y = SARImgParams.Wy_m / (SARImgParams.N_y_pix - 1);
    //std::cout << "(minRvec,maxRvec) = (" << min_Rvec << ", " << max_Rvec << ")" << std::endl;
    for (int xIdx = 0; xIdx < SARImgParams.N_x_pix; xIdx++) {
        target.y = SARImgParams.y0_m - (SARImgParams.Wy_m / 2);
        for (int yIdx = 0; yIdx < SARImgParams.N_y_pix; yIdx++) {
            float dR_val = std::sqrt((SARData.Ant_x.data[pulseIndex] - target.x) * (SARData.Ant_x.data[pulseIndex] - target.x) +
                    (SARData.Ant_y.data[pulseIndex] - target.y) * (SARData.Ant_y.data[pulseIndex] - target.y) +
                    (SARData.Ant_z.data[pulseIndex] - target.z) * (SARData.Ant_z.data[pulseIndex] - target.z)) - SARData.slant_range.data[pulseIndex];
            //  std::cout << "y= " << target.y << " dR(" << xIdx << ", " << yIdx << ") = " << dR_val << std::endl;
            if (dR_val > min_Rvec && dR_val < max_Rvec) {
                // TODO: Amiguate as default double and have it cast to float if appropriate for precision specifications
                Complex phCorr_val = polarToComplex(1.0f, (__nTpData) ((4.0 * PI * SARData.startF.data[pulseIndex] * dR_val) / CLIGHT));
                //std::cout << "idx = " << (xIdx * Ny_pix + yIdx) << " (x,y)=(" << target.x << "," << target.y << ")"
                //        << "(dR,phCorr)=(" << dR_val << ", " << phCorr_val << ")" << std::endl;
                __nTpData dR_idx = RANGE_VALUE_TO_RANGE_INDEX(dR_val, SARImgParams.max_Wy_m, SARImgParams.N_fft);
                Complex iRC_val = interp1(r_vec, SARImgParams.N_fft, rangeCompressed, dR_val, dR_idx);
                //int outputIdx = xIdx * SARImgParams.N_y_pix + yIdx;
                //std::cout << "output[" << outputIdx << "] += " << (iRC_val * phCorr_val) << std::endl;
                output_image[xIdx * SARImgParams.N_y_pix + yIdx] += iRC_val * phCorr_val;
            }
            target.y += delta_y;
        }
        target.x += delta_x;
    }
}

template <typename __nTpData, typename __nTpParams>
void run_bp(const SAR_Aperture<__nTpData>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        CArray& output_image) {

    std::cout << "Running backprojection SAR focusing algorithm." << std::endl;
    /*
        % Calculate the range to every bin in the range profile (m)
        data.r_vec = linspace(-data.Nfft/2,data.Nfft/2-1,data.Nfft)*data.maxWr/data.Nfft;
     */

    // TODO: Add range vector to SARData structure
    __nTpData r_vec[SARImgParams.N_fft];
    __nTpData min_Rvec = std::numeric_limits<float>::infinity();
    __nTpData max_Rvec = -std::numeric_limits<float>::infinity();
    for (int rIdx = 0; rIdx < SARImgParams.N_fft; rIdx++) {
        // -maxWr/2:maxWr/Nfft:maxWr/2
        //float rVal = ((float) rIdx / Nfft - 0.5f) * maxWr;
        __nTpData rVal = RANGE_INDEX_TO_RANGE_VALUE(rIdx, SARImgParams.max_Wy_m, SARImgParams.N_fft);
        r_vec[rIdx] = rVal;
        if (min_Rvec > r_vec[rIdx]) {
            min_Rvec = r_vec[rIdx];
        }
        if (max_Rvec < r_vec[rIdx]) {
            max_Rvec = r_vec[rIdx];
        }
    }

    __nTpData timeleft = 0.0f;

    const Complex* range_profiles_cast = static_cast<const Complex*> (&SARData.sampleData.data[0]);
    //mxComplexSingleClass* output_image_cast = static_cast<mxComplexSingleClass*> (output_image);

    CArray range_profiles_arr(range_profiles_cast, SARData.numAzimuthSamples * SARData.numRangeSamples);

    for (int pulseIndex = 0; pulseIndex < SARData.numAzimuthSamples; pulseIndex++) {
        if (pulseIndex > 1 && (pulseIndex % 100) == 0) {
            std::cout << "Pulse " << pulseIndex << " of " << SARData.numAzimuthSamples
                    << ", " << std::setprecision(2) << timeleft << " minutes remaining" << std::endl;
        }

        CArray phaseData = range_profiles_arr[std::slice(pulseIndex * SARImgParams.N_fft, SARImgParams.N_fft, 1)];

        //ifft(phaseData);
        ifftw(phaseData);

        CArray rangeCompressed = fftshift(phaseData);
        computeDifferentialRangeAndPhaseCorrections(pulseIndex, SARData,
                SARImgParams, rangeCompressed,
                r_vec, min_Rvec, max_Rvec,
                output_image);
    }
}

template <typename __nTpData, typename __nTpParams>
void computeDifferentialRangeAndPhaseCorrectionsMF(int pulseIndex,
        const SAR_Aperture<__nTpData>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        CArray& output_image) {

    float4 target;
    target.z = 0;
    float delta_x = SARImgParams.Wx_m / (SARImgParams.N_x_pix - 1);
    float delta_y = SARImgParams.Wy_m / (SARImgParams.N_y_pix - 1);
    //std::cout << "(minRvec,maxRvec) = (" << min_Rvec << ", " << max_Rvec << ")" << std::endl;
    int pulse_startF_FreqIdx = pulseIndex * SARData.numRangeSamples;
    for (int freqIdx = 0; freqIdx < SARData.numRangeSamples; freqIdx++) {
        // TODO: Amiguate as default double and have it cast to float if appropriate for precision specifications
        const Complex& phaseHistorySample = SARData.sampleData.data[pulse_startF_FreqIdx + freqIdx];
        target.x = SARImgParams.x0_m - (SARImgParams.Wx_m / 2);
        for (int xIdx = 0; xIdx < SARImgParams.N_x_pix; xIdx++) {
            target.y = SARImgParams.y0_m - (SARImgParams.Wy_m / 2);
            for (int yIdx = 0; yIdx < SARImgParams.N_y_pix; yIdx++) {
                float dR_val = std::sqrt((SARData.Ant_x.data[pulseIndex] - target.x) * (SARData.Ant_x.data[pulseIndex] - target.x) +
                        (SARData.Ant_y.data[pulseIndex] - target.y) * (SARData.Ant_y.data[pulseIndex] - target.y) +
                        (SARData.Ant_z.data[pulseIndex] - target.z) * (SARData.Ant_z.data[pulseIndex] - target.z)) - SARData.slant_range.data[pulseIndex];
                //  std::cout << "y= " << target.y << " dR(" << xIdx << ", " << yIdx << ") = " << dR_val << std::endl;
                //int outputIdx = xIdx * SARImgParams.N_y_pix + yIdx;
                //std::cout << "output[" << outputIdx << "] += " << (iRC_val * phCorr_val) << std::endl;
                Complex phCorr_val = polarToComplex(1.0f, (__nTpData) ((4.0 * PI * SARData.freq.data[pulse_startF_FreqIdx + freqIdx] * dR_val) / CLIGHT));
                output_image[xIdx * SARImgParams.N_y_pix + yIdx] += phaseHistorySample * phCorr_val;
                target.y += delta_y;
            }
            target.x += delta_x;
        }
    }
}

template <typename __nTpData, typename __nTpParams>
void run_mf(const SAR_Aperture<__nTpData>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        CArray& output_image) {

    std::cout << "Running matched filter SAR focusing algorithm." << std::endl;
    __nTpData timeleft = 0.0f;

    const Complex* range_profiles_cast = static_cast<const Complex*> (&SARData.sampleData.data[0]);
    //mxComplexSingleClass* output_image_cast = static_cast<mxComplexSingleClass*> (output_image);

    CArray range_profiles_arr(range_profiles_cast, SARData.numAzimuthSamples * SARData.numRangeSamples);

    for (int pulseIndex = 0; pulseIndex < SARData.numAzimuthSamples; pulseIndex++) {
        if (pulseIndex > 1) {// && (pulseIndex % 100) == 0) {
            std::cout << "Pulse " << pulseIndex << " of " << SARData.numAzimuthSamples
                    << ", " << std::setprecision(2) << timeleft << " minutes remaining" << std::endl;

            computeDifferentialRangeAndPhaseCorrectionsMF(pulseIndex, SARData,
                    SARImgParams, output_image);
        }
    }
}

template <typename __nTpData, typename __nTpParams>
void focus_SAR_image(const SAR_Aperture<__nTpData>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        CArray & output_image) {

    // Display maximum scene size and resolution
    std::cout << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << SARImgParams.max_Wy_m << " m range, "
            << SARImgParams.max_Wx_m << " m cross-range" << std::endl;
    std::cout << "Resolution:  " << std::fixed << std::setprecision(2) << SARImgParams.slant_rangeResolution << "m range, "
            << SARImgParams.azimuthResolution << " m cross-range" << std::endl;

    run_mf(SARData, SARImgParams, output_image);
}

#endif /* CPUBACKPROJECTION_HPP */

