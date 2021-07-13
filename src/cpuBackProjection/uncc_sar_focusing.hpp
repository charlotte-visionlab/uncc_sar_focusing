/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   uncc_sar_focusing.hpp
 * Author: arwillis
 *
 * Created on June 20, 2021, 9:46 AM
 */

#ifndef UNCC_SAR_FOCUSING_HPP
#define UNCC_SAR_FOCUSING_HPP

#include <complex>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <valarray>
#include <vector>

#include "uncc_complex.hpp"

#define CLIGHT 299792458.0 /* c: speed of light, m/s */
//#define CLIGHT 299792458.0f /* c: speed of light, m/s */
#define PI 3.141592653589793   /* pi, accurate to 6th place in single precision */
//#define PI 3.14159265359f   /* pi, accurate to 6th place in single precision */

template<typename __nTp>
using Complex = unccComplex<__nTp>;

//template<typename __nTp> 
//using Complex = std::complex<__nTp>;

template<typename __nTp>
using CArray = std::valarray<Complex<__nTp> >;

template<typename __numTp>
struct SimpleMatrix {
public:
    std::vector<int> shape;
    std::vector<__numTp> data;
    typedef std::shared_ptr<SimpleMatrix> Ptr;

    int nelem() {
        return (shape.size() == 0) ? 0 :
                std::accumulate(begin(shape), end(shape), 1, std::multiplies<int>()); // / (polarityIdx == -1 ? 1 : shape[polarityIdx]);
    }

    bool isEmpty() {
        return shape.size() == 0;
    }

    SimpleMatrix getDataChannel(int channel_index, int channel_number) {
        SimpleMatrix<__numTp> output;
        int elem_per_channel = 1;
        for (int cidx = 0; cidx < channel_index; cidx++) {
            elem_per_channel *= shape[cidx];
            output.shape.push_back(shape[cidx]);
        }
        int start_offset = elem_per_channel * channel_number;
//        output.data.insert(data.begin() + start_offset,
//                data.begin() + start_offset + elem_per_channel,
//                std::back_inserter(output.data));
        return output;
    }

    SimpleMatrix::Ptr create() {
        return std::make_shared<SimpleMatrix>();
    }

    template <typename _Tp>
    friend std::ostream& operator<<(std::ostream& output, const SimpleMatrix<__numTp> &c);

};

template <typename __numTp>
inline std::ostream& operator<<(std::ostream& output, const SimpleMatrix<__numTp>& sMat) {
    int MAX_NUM_VARS_PRINTED = 10;
    std::string dimsStr;
    if (!sMat.shape.empty()) {
        dimsStr = std::accumulate(sMat.shape.begin() + 1, sMat.shape.end(),
                std::to_string(sMat.shape[0]), [](const std::string & a, int b) {
                    return a + ',' + std::to_string(b);
                });
    }
    output << "[" << dimsStr << "] = {"; \
    typename std::vector<__numTp>::const_iterator i;
    for (i = sMat.data.begin(); i != sMat.data.end() && i != sMat.data.begin() + MAX_NUM_VARS_PRINTED; ++i) {
        output << *i << ", ";
    }
    output << ((sMat.data.size() > MAX_NUM_VARS_PRINTED) ? " ..." : "") << " }";
    return output;
}

// For back projection algorithm 

template <typename __Tp>
class RangeBinData {
public:
    typedef std::shared_ptr<RangeBinData> Ptr;
    // rangeBins for backprojection / compressed range processing
    SimpleMatrix<__Tp> rangeBins;
    __Tp minRange;
    __Tp maxRange;
};

template<typename _numTp>
class SAR_Aperture {
public:
    typedef std::shared_ptr<SAR_Aperture> Ptr;

    bool format_GOTCHA;

    // GOTCHA + Sandia Fields
    SimpleMatrix<Complex<_numTp>> sampleData;
    //int numPulses;
    //int numRangeSamples;
    SimpleMatrix<_numTp> Ant_x;
    SimpleMatrix<_numTp> Ant_y;
    SimpleMatrix<_numTp> Ant_z;

    // GOTCHA-Only Fields
    SimpleMatrix<_numTp> freq;
    SimpleMatrix<_numTp> slant_range;
    SimpleMatrix<_numTp> theta;
    SimpleMatrix<_numTp> phi;

    struct {
        SimpleMatrix<_numTp> r_correct;
        SimpleMatrix<_numTp> ph_correct;
    } af;

    // Sandia-ONLY Fields
    SimpleMatrix<_numTp> ADF;
    SimpleMatrix<_numTp> startF;
    SimpleMatrix<_numTp> chirpRate;
    SimpleMatrix<_numTp> chirpRateDelta;

    // Fields set automatically by program computations or manually via user input arguments
    // 1 - HH, 2 -HV, 3 - VH, 4 - VV
    int polarity_channel;
    // array index to use for polarity data when indexing multi-dimensional arrays
    // -1 = there is only one polarity in the SAR data file
    int polarity_dimension;

    // Fields below set automatically
    int numRangeSamples;
    int numAzimuthSamples;
    int numPolarities;
    SimpleMatrix<_numTp> bandwidth;
    SimpleMatrix<_numTp> deltaF;

    _numTp mean_startF;
    _numTp mean_deltaF;
    _numTp mean_bandwidth;

    SimpleMatrix<_numTp> Ant_Az;
    SimpleMatrix<_numTp> Ant_deltaAz;
    SimpleMatrix<_numTp> Ant_El;
    SimpleMatrix<_numTp> Ant_deltaEl;
    _numTp mean_Ant_El;
    _numTp mean_deltaAz;
    _numTp mean_deltaEl;
    _numTp mean_Ant_deltaAz;
    _numTp mean_Ant_deltaEl;
    _numTp Ant_totalAz;
    _numTp Ant_totalEl;

    SAR_Aperture() : format_GOTCHA(true), polarity_channel(1), polarity_dimension(-1) {
    };

    virtual ~SAR_Aperture() {
    };

    SAR_Aperture::Ptr create() {
        return std::make_shared<SAR_Aperture>();
    }

    template <typename _Tp>
    friend std::ostream& operator<<(std::ostream& output, const SAR_Aperture<_Tp> &c);

    int exportData(SAR_Aperture<_numTp>& dstData, int data_channel) {
        dstData.numRangeSamples = sampleData.shape[0];
        dstData.numAzimuthSamples = sampleData.shape[1];
        dstData.numPolarities = (sampleData.shape.size() > 2) ? sampleData.shape[2] : 1;
        dstData.sampleData = sampleData.getDataChannel(2, 1);
        return EXIT_SUCCESS;
    }

};

template <typename _numTp>
inline std::ostream& operator<<(std::ostream& output, const SAR_Aperture<_numTp>& c) {
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
        output << "ChirpRate" << c.chirpRate << std::endl;
        output << "ChirpRateDelta" << c.chirpRateDelta << std::endl;
    }
    return output;
}

#define numValuesPerPolarity(sMat, pDim) sMat.nelem()/((sMat.shape.size() >= pDim && pDim != -1) ? sMat.shape[pDim] : 1)

// TODO: Make this a class function for SAR_Aperture

template<typename _numTp>
int initialize_SAR_Aperture_Data(SAR_Aperture<_numTp>& aperture) {

    aperture.numRangeSamples = aperture.sampleData.shape[0];
    aperture.numAzimuthSamples = aperture.sampleData.shape[1];
    aperture.numPolarities = (aperture.sampleData.shape.size() > 2) ? aperture.sampleData.shape[2] : 1;

    int numSARSamples = aperture.numRangeSamples * aperture.numAzimuthSamples;
    const int polDim = aperture.polarity_dimension;
    // determine if there are sufficient antenna phase center values to focus the SAR image data
    if (numValuesPerPolarity(aperture.Ant_x, polDim) != aperture.numAzimuthSamples ||
            numValuesPerPolarity(aperture.Ant_y, polDim) != aperture.numAzimuthSamples ||
            numValuesPerPolarity(aperture.Ant_z, polDim) != aperture.numAzimuthSamples) {
        std::cout << "initializeSARFocusingVariables::Not enough antenna positions available to focus the selected SAR data." << std::endl;
        return EXIT_FAILURE;
    }

    // populate frequency sample locations for every pulse if not already available
    // also populates startF and deltaF in some cases
    if (numValuesPerPolarity(aperture.freq, polDim) != numSARSamples) {
        std::cout << "initializeSARFocusingVariables::Found " << numValuesPerPolarity(aperture.freq, polDim)
                << " frequency measurements and need " << numSARSamples << " measurements. Augmenting frequency data for SAR focusing." << std::endl;
        if (!aperture.freq.isEmpty() && aperture.freq.shape[0] == aperture.numRangeSamples) {
            std::cout << "Assuming constant frequency samples for each SAR pulse." << std::endl;
            // make aperture.numAzimuthSamples-1 copies of the first frequency sample vector
            aperture.freq.shape.clear();
            aperture.freq.shape.push_back(aperture.numRangeSamples);
            aperture.freq.shape.push_back(aperture.numAzimuthSamples);
            _numTp minFreq = *std::min_element(std::begin(aperture.freq.data), std::end(aperture.freq.data));
            _numTp maxFreq = *std::max_element(std::begin(aperture.freq.data), std::end(aperture.freq.data));
            _numTp bandwidth = maxFreq - minFreq;
            _numTp deltaF = std::abs(aperture.freq.data[1] - aperture.freq.data[0]);
            bool fill_startF = false;
            if (numValuesPerPolarity(aperture.startF, polDim) != aperture.numAzimuthSamples) {
                fill_startF = true;
                aperture.startF.shape.push_back(aperture.numAzimuthSamples);
            }
            bool fill_deltaF = false;
            if (numValuesPerPolarity(aperture.deltaF, polDim) != aperture.numAzimuthSamples) {
                fill_deltaF = true;
                aperture.deltaF.shape.push_back(aperture.numAzimuthSamples);
            }
            aperture.bandwidth.shape.push_back(aperture.numAzimuthSamples);
            for (int azIdx = 0; azIdx < aperture.numAzimuthSamples; ++azIdx) {
                // assume we already have one set of frequency sample values
                // then we have to make aperture.numAzimuthSamples-1 copies
                if (azIdx > 0) {
                    aperture.freq.data.insert(aperture.freq.data.end(), &aperture.freq.data[0], &aperture.freq.data[aperture.numRangeSamples]);
                }
                aperture.bandwidth.data.push_back(bandwidth);
                if (fill_startF) {
                    aperture.startF.data.push_back(minFreq);
                }
                if (fill_deltaF) {
                    aperture.deltaF.data.push_back(deltaF);
                }
            }
        } else if (!aperture.startF.isEmpty() && aperture.startF.shape[1] == aperture.numAzimuthSamples &&
                !aperture.ADF.isEmpty() && aperture.ADF.shape[0] == 1 &&
                !aperture.chirpRate.isEmpty() && aperture.chirpRate.shape[1] == aperture.numAzimuthSamples) {
            std::cout << "Assuming variable frequency samples for each SAR pulse. Interpolating frequency samples from chirp rate, sample rate and start frequency." << std::endl;
            aperture.deltaF.shape.clear();
            aperture.deltaF.data.clear();
            aperture.deltaF.shape.push_back(aperture.numAzimuthSamples);
            aperture.bandwidth.shape.clear();
            aperture.bandwidth.data.clear();
            aperture.bandwidth.shape.push_back(aperture.numAzimuthSamples);
            aperture.freq.shape.clear();
            aperture.freq.data.clear();
            aperture.freq.shape.push_back(aperture.numRangeSamples);
            aperture.freq.shape.push_back(aperture.numAzimuthSamples);
            for (int azIdx = 0; azIdx < aperture.numAzimuthSamples; ++azIdx) {
                for (int freqIdx = 0; freqIdx < aperture.numRangeSamples; freqIdx++) {
                    _numTp freqSample = aperture.startF.data[azIdx] + freqIdx * aperture.chirpRate.data[azIdx] / aperture.ADF.data[0];
                    aperture.freq.data.push_back(freqSample);
                }
                //_numTp minFreq = aperture.startF.data[azIdx];
                _numTp deltaF = aperture.chirpRate.data[azIdx] / aperture.ADF.data[0];
                aperture.deltaF.data.push_back(deltaF);
                _numTp bandwidth = ((aperture.numRangeSamples - 1) * aperture.chirpRate.data[azIdx]) / aperture.ADF.data[0];
                aperture.bandwidth.data.push_back(bandwidth);
            }
        }
    }

    // populate slant_range to target phase center for every pulse if not already available
    if (numValuesPerPolarity(aperture.slant_range, polDim) != aperture.numAzimuthSamples) {
        std::cout << "initializeSARFocusingVariables::Found " << numValuesPerPolarity(aperture.slant_range, polDim)
                << " slant range measurements and need " << aperture.numAzimuthSamples << " measurements. Augmenting slant range data for SAR focusing." << std::endl;
        aperture.slant_range.shape.clear();
        aperture.slant_range.data.clear();
        aperture.slant_range.shape.push_back(aperture.numAzimuthSamples);
        for (int azIdx = 0; azIdx < aperture.numAzimuthSamples; ++azIdx) {
            aperture.slant_range.data.push_back(std::sqrt((aperture.Ant_x.data[azIdx] * aperture.Ant_x.data[azIdx]) +
                    (aperture.Ant_y.data[azIdx] * aperture.Ant_y.data[azIdx]) +
                    (aperture.Ant_z.data[azIdx] * aperture.Ant_z.data[azIdx])));
        }
    }

    // populate deltaF if not already available
    if (numValuesPerPolarity(aperture.deltaF, polDim) != aperture.numAzimuthSamples) {
        _numTp deltaF = std::abs(aperture.freq.data[1] - aperture.freq.data[0]);
        for (int azIdx = 0; azIdx < aperture.numAzimuthSamples; ++azIdx) {
            for (int freqIdx = 0; freqIdx < aperture.numRangeSamples; freqIdx++) {
                // TODO: polarity index for sourcing data
                int sampleIndex = azIdx * aperture.numRangeSamples + freqIdx;
                _numTp deltaF = std::abs(aperture.freq.data[sampleIndex + 1] - aperture.freq.data[sampleIndex]);
                aperture.deltaF.data.push_back(deltaF);
            }
        }
    }

    // calculate frequency statistics
    _numTp sum_startF = std::accumulate(aperture.startF.data.begin(), aperture.startF.data.end(), 0.0);
    aperture.mean_startF = sum_startF / aperture.startF.data.size();

    _numTp sum_deltaF = std::accumulate(aperture.deltaF.data.begin(), aperture.deltaF.data.end(), 0.0);
    aperture.mean_deltaF = sum_deltaF / aperture.deltaF.data.size();

    _numTp sum_bandwidth = std::accumulate(aperture.bandwidth.data.begin(), aperture.bandwidth.data.end(), 0.0);
    aperture.mean_bandwidth = sum_bandwidth / aperture.bandwidth.data.size();

    if (numValuesPerPolarity(aperture.Ant_Az, polDim) != aperture.numAzimuthSamples) {
        aperture.Ant_Az.shape.clear();
        aperture.Ant_Az.data.clear();
        aperture.Ant_Az.shape.push_back(aperture.numAzimuthSamples);
        aperture.Ant_El.shape.clear();
        aperture.Ant_El.data.clear();
        aperture.Ant_El.shape.push_back(aperture.numAzimuthSamples);
        aperture.Ant_deltaAz.shape.clear();
        aperture.Ant_deltaAz.data.clear();
        aperture.Ant_deltaAz.shape.push_back(aperture.numAzimuthSamples - 1);
        aperture.Ant_deltaEl.shape.clear();
        aperture.Ant_deltaEl.data.clear();
        aperture.Ant_deltaEl.shape.push_back(aperture.numAzimuthSamples - 1);
        for (int azIdx = 0; azIdx < aperture.numAzimuthSamples; ++azIdx) {
            // TODO: polarity index for sourcing data
            int sampleIndex = azIdx;
            // TODO: unwrap the azimuth for spotlight SAR that crosses the 2*PI boundary
            aperture.Ant_Az.data.push_back(std::atan2(aperture.Ant_y.data[sampleIndex], aperture.Ant_x.data[sampleIndex]));
            _numTp Ant_groundRange_to_phaseCenter = std::sqrt((aperture.Ant_x.data[azIdx] * aperture.Ant_x.data[azIdx]) +
                    (aperture.Ant_y.data[azIdx] * aperture.Ant_y.data[azIdx]));
            aperture.Ant_El.data.push_back(std::atan2(aperture.Ant_z.data[sampleIndex], Ant_groundRange_to_phaseCenter));
            if (azIdx > 0) {
                aperture.Ant_deltaAz.data.push_back(aperture.Ant_Az.data[sampleIndex] - aperture.Ant_Az.data[sampleIndex - 1]);
                aperture.Ant_deltaEl.data.push_back(aperture.Ant_El.data[sampleIndex] - aperture.Ant_El.data[sampleIndex - 1]);
            }
        }
        _numTp sum_Ant_deltaAz = std::accumulate(aperture.Ant_deltaAz.data.begin(), aperture.Ant_deltaAz.data.end(), 0.0);
        aperture.mean_Ant_deltaAz = sum_Ant_deltaAz / aperture.Ant_deltaAz.data.size();

        _numTp sum_Ant_El = std::accumulate(aperture.Ant_El.data.begin(), aperture.Ant_El.data.end(), 0.0);
        aperture.mean_Ant_El = sum_Ant_El / aperture.Ant_El.data.size();

        _numTp sum_Ant_deltaEl = std::accumulate(aperture.Ant_deltaEl.data.begin(), aperture.Ant_deltaEl.data.end(), 0.0);
        aperture.mean_Ant_deltaEl = sum_Ant_deltaEl / aperture.Ant_deltaEl.data.size();

        _numTp min_Ant_Az = *std::min_element(std::begin(aperture.Ant_Az.data), std::end(aperture.Ant_Az.data));
        _numTp max_Ant_Az = *std::max_element(std::begin(aperture.Ant_Az.data), std::end(aperture.Ant_Az.data));
        aperture.Ant_totalAz = max_Ant_Az - min_Ant_Az;

        _numTp min_Ant_El = *std::min_element(std::begin(aperture.Ant_El.data), std::end(aperture.Ant_El.data));
        _numTp max_Ant_El = *std::max_element(std::begin(aperture.Ant_El.data), std::end(aperture.Ant_El.data));
        aperture.Ant_totalEl = max_Ant_El - min_Ant_El;
    }

    return EXIT_SUCCESS;
}

template<typename _numTp>
class SAR_ImageFormationParameters {
public:

    enum ALGORITHM {
        UNKNOWN,
        MATCHED_FILTER,
        BACKPROJECTION
    } algorithm;
    int pol; // What polarization to image (HH,HV,VH,VV)
    // Define image parameters here
    int N_fft; // Number of samples in FFT
    bool zeropad_fft;
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

    CUDAFUNCTION SAR_ImageFormationParameters() : N_fft(512), N_x_pix(512), N_y_pix(512),
    x0_m(0), y0_m(0), dyn_range_dB(70), zeropad_fft(true), algorithm(ALGORITHM::BACKPROJECTION) {
    };

    template <typename __argTp>
    CUDAFUNCTION void update(const SAR_Aperture<__argTp> aperture) {
        // Determine the maximum scene size of the image (m)
        // max down-range/fast-time/y-axis extent of image (m)
        max_Wy_m = CLIGHT / (2.0 * aperture.mean_deltaF);
        // max cross-range/fast-time/x-axis extent of image (m)
        max_Wx_m = CLIGHT / (2.0 * std::abs(aperture.mean_Ant_deltaAz) * aperture.mean_startF);
        // Determine the resolution of the image (m)
        slant_rangeResolution = CLIGHT / (2.0 * aperture.mean_bandwidth);
        ground_rangeResolution = slant_rangeResolution / std::sin(aperture.mean_Ant_El);
        azimuthResolution = CLIGHT / (2.0 * aperture.Ant_totalAz * aperture.mean_startF);
    }

    template <typename __myTp, typename __argTp>
    CUDAFUNCTION static SAR_ImageFormationParameters create(const SAR_Aperture<__argTp> aperture) {
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

    CUDAFUNCTION ~SAR_ImageFormationParameters() {
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
    return output;
}

template<typename __nTp>
void fft(CArray<__nTp>& x);

template<typename __nTp>
void ifft(CArray<__nTp>& x);

template<typename __nTp>
void fftw(CArray<__nTp>& x);

template<typename __nTp>
void ifftw(CArray<__nTp>& x);

template<typename __nTp>
CArray<__nTp> fftshift(CArray<__nTp>& fft);

template <typename __nTp, typename __nTpParams>
void focus_SAR_image(const SAR_Aperture<__nTp>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        CArray<__nTp>& output_image);

#endif /* UNCC_SAR_FOCUSING_HPP */

