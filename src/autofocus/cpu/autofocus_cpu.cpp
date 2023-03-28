/*
 * Copyright (C) 2022 Andrew R. Willis
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

// Standard Library includes
#include <iomanip>
#include <sstream>
#include <fstream>

#include <third_party/log.h>
#include <third_party/cxxopts.hpp>

// this declaration needs to be in any C++ compiled target for CPU
#define CUDAFUNCTION

#include <charlotte_sar_api.hpp>
#include <uncc_sar_globals.hpp>

#include <uncc_sar_focusing.hpp>
#include <uncc_sar_matio.hpp>

#include "../../cpuBackProjection/cpuBackProjection.hpp"

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <Eigen/Dense>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

typedef float NumericType;

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor {
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix <Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}

    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }

    int values() const { return m_values; }

};

template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct PGAFunctor {
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix <Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    int m_inputs, m_values, numSamples, numRange;

    PGAFunctor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}

    PGAFunctor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }

    int values() const { return m_values; }

    int operator()(const Eigen::VectorXcf &x, Eigen::VectorXf &fvec) const {
        // Implement the ()
        // Energy of the phase, all of the phase given the phase shift
        // Python code up to phi (remove linear trend), try it with and without

//        cufftComplex *G_dot = new cufftComplex[numSamples * numRange]; // Holds the data difference
//        float *phi_dot = new float[numSamples]; // Holds phi_dot and will also hold phi for simplicity// Store values into G_dot
//
//        for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
//            for (int rangeNum = 0; rangeNum < numRange - 1; rangeNum++) { // Because it's a difference
//                G_dot[rangeNum + pulseNum * numRange]._M_real =
//                        x(rangeNum + pulseNum * numRange).real() - x(rangeNum + pulseNum * numRange + 1).real();
//                G_dot[rangeNum + pulseNum * numRange]._M_imag =
//                        x(rangeNum + pulseNum * numRange).imag() - x(rangeNum + pulseNum * numRange + 1).imag();
//            }
//            // To follow the python code where they append the final sample difference to the matrix to make the size the same as the original
//            G_dot[numRange - 1 + pulseNum * numRange]._M_real = G_dot[numRange - 2 + pulseNum * numRange]._M_real;
//            G_dot[numRange - 1 + pulseNum * numRange]._M_imag = G_dot[numRange - 2 + pulseNum * numRange]._M_imag;
//        }
//
//        for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
//            float G_norm = 0; // Something to temporarily hold the summed data Norm for that sample
//            for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
//                int idx = rangeNum + pulseNum * numRange;
//                phi_dot[pulseNum] += (x(idx).real() * G_dot[idx]._M_imag) +
//                                     (-1 * x(idx).imag() * G_dot[idx]._M_real); // Only the imaginary component is needed
//                G_norm += sqrt(x(idx).real() * x(idx).real() + x(idx).imag() * x(idx).imag());
//            }
//            phi_dot[pulseNum] /= G_norm;
//        }
//
//        for (int pulseNum = 1; pulseNum < numSamples; pulseNum++) { // Integrate to get phi
//            phi_dot[pulseNum] = phi_dot[pulseNum] + phi_dot[pulseNum - 1];
//        }
//
//        delete[] G_dot;
//        delete[] phi_dot;
        return 0;
    }

    int df(const Eigen::VectorXcf &x, Eigen::MatrixXf &fjac) const {
        for (int iii = 0; iii < x.size(); iii++) {
            // Still need to figure out how x will look like (Array of complex vectors?)
            float b = x(iii).imag(); // Something like this, .y or .imag()
            fjac(0, iii) = b * b;
        }
        return 0;
    }
};

template<typename __nTp>
void bestFit(__nTp *coeffs, std::vector<__nTp> values, int nPulse, int skip) {
    // double sumX = 0.0;
    // double sumY = 0.0;
    // double N = values.size();
    // double sumXY = 0.0;
    // double sumXX = 0.0;

    // for (int i =0; i < values.size(); i++) {
    //     sumX += (__nTp)i;
    //     sumY += values[i];
    //     sumXY += ((__nTp)i*values[i]);
    //     sumXX += ((__nTp)i * (__nTp)i);
    // }

    // double numS = N * sumXY - sumX * sumY;
    // double den = N * sumXX - sumX * sumX;

    // double numC = sumY * sumXX - sumX * sumXY;

    // double temp1 = numS/den;
    // double temp2 = numC/den;

    // float tempa = (float)temp1;
    // float tempb = (float)temp2;

    // coeffs[0] = 0;
    // coeffs[1] = tempa;
    // coeffs[2] = tempb;

    int numN = nPulse;
    double N = 0;
    double x1 = 0;
    double x2 = 0;
    double f0 = 0;
    double f1 = 0;

    for (int i = 0; i < numN; i += skip) {
        N += 1;
        x1 += i;
        x2 += i * i;
        f0 += values[i];
        f1 += values[i] * i;
    }

    double D = -1 * x1 * x1 + N * x2;

    double a = N * f1 - f0 * x1;
    double b = f0 * x2 - f1 * x1;
    // printf("N = %f\nx1 = %f\nx2 = %f\nf0 = %f\nf1 = %f\n", N, x1, x2, f0, f1);
    // printf("a = %f\nb = %f\n D = %f\n", a, b, D);
    a /= D;
    b /= D;
    // printf("a1 = %f\nb2 = %f\n", a, b);
    float temp_a = (float) a;
    float temp_b = (float) b;
    // printf("temp_a = %f\ntemp_b = %f\n", temp_a, temp_b);
    coeffs[0] = temp_b;
    coeffs[1] = temp_a;
}

template<typename __nTp>
void quadFit(__nTp *coeffs, std::vector<__nTp> values, int nPulse, int skip) {
    int numN = nPulse;
    double N = 0;
    double x1 = 0;
    double x2 = 0;
    double x3 = 0;
    double x4 = 0;
    double f0 = 0;
    double f1 = 0;
    double f2 = 0;

    for (int i = 0; i < numN; i += skip) {
        N += 1;
        x1 += i;
        x2 += i * i;
        x3 += i * i * i;
        x4 += i * i * i * i;
        f0 += values[i];
        f1 += values[i] * i;
        f2 += values[i] * i * i;
    }

    double D = x4 * x1 * x1 - 2 * x1 * x2 * x3 + x2 * x2 * x2 - N * x4 * x2 + N * x3 * x3;

    double a = f2 * x1 * x1 - f1 * x1 * x2 - f0 * x3 * x1 + f0 * x2 * x2 - N * f2 * x2 + N * f1 * x3;
    double b = f1 * x2 * x2 - N * f1 * x4 + N * f2 * x3 + f0 * x1 * x4 - f0 * x2 * x3 - f2 * x1 * x2;
    double c = f2 * x2 * x2 - f1 * x2 * x3 - f0 * x4 * x2 + f0 * x3 * x3 - f2 * x1 * x3 + f1 * x1 * x4;

    a /= D;
    b /= D;
    c /= D;

    float temp_a = (float) a;
    float temp_b = (float) b;
    float temp_c = (float) c;

    coeffs[0] = temp_c;
    coeffs[1] = temp_b;
    coeffs[2] = temp_a;
}

template<typename __nTp>
CArray<__nTp> fftunshift(CArray<__nTp>& fft) {
    return fft.cshift(-1*(fft.size() + 1) / 2);
}

template<typename __nTp>
void autofocus(Complex<__nTp> *data, int numSamples, int numRange, int numIterations) {
    // TODO: ADD ifft/shift before G_dot stuff
    // Samples are number of columns, range is the number of rows
    // Data is column-major ordered
    // CArray<__nTp> G_dot(numSamples * numRange);
    Complex<__nTp> *G = new Complex<__nTp>[numSamples * numRange];
    Complex<__nTp> *G_dot = new Complex<__nTp>[numSamples * numRange]; // Holds the data difference
    double *phi_dot = new double[numSamples]; // Holds phi_dot and will also hold phi for simplicity

    int iii = 0;
    for (; iii < numIterations; iii++) {
        for(int j = 0; j < numSamples; j++) {
            phi_dot[j] = 0;
        }
        CArray<__nTp>temp_g(data, numSamples*numRange);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Shifting the max values
// For each row
        for (int rangeIndex = 0; rangeIndex < numRange; rangeIndex++) {
            CArray<__nTp> phaseData = temp_g[std::slice(rangeIndex, numSamples, numRange)];
            CArray<__nTp> tempData = temp_g[std::slice(rangeIndex, numSamples, numRange)];

            // Need to shift phaseData around so max is in the beginning
            int maxIdx = 0;
            __nTp maxVal = 0;
            for (int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
                __nTp tempVal = Complex<__nTp>::abs(phaseData[pulseIndex]);
                if(tempVal > maxVal) {
                    maxVal = tempVal;
                    maxIdx = pulseIndex;
                }
            }

            phaseData = phaseData.cshift(phaseData.size() / 2 + maxIdx);

            for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
                temp_g[rangeIndex + pulseIndex * numRange] = phaseData[pulseIndex];
            }
        }

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Windowing
        double avg_window[numSamples] = {0};
        for (int rangeIndex = 0; rangeIndex < numRange; rangeIndex++) {
            for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
                int idx = rangeIndex + pulseIndex * numRange;
                Complex<__nTp> tempHolder = Complex<__nTp>::conj(data[idx]) * data[idx];
                avg_window[pulseIndex] += tempHolder.real();
            }
        }

        double window_maxVal = 0;
        for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
            if(window_maxVal < avg_window[pulseIndex]) window_maxVal = avg_window[pulseIndex];
        }

        for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
            avg_window[pulseIndex] = 10 * log10(avg_window[pulseIndex] / window_maxVal);
        }

        int leftIdx = -1;
        int rightIdx = -1;
        for(int i = 0; i < numSamples / 2; i++) {
            if(avg_window[i] < -30) leftIdx = i;
            if(avg_window[i + numSamples / 2] < -30 && rightIdx == -1) rightIdx = i;
        }

        if (leftIdx == -1) leftIdx = 0;
        if (rightIdx == -1) rightIdx = numSamples;
        
        Complex<__nTp> tempZero(0, 0);
        for (int rangeIndex = 0; rangeIndex < numRange; rangeIndex++) {
            for(int pulseIndex = 0; pulseIndex < leftIdx; pulseIndex++) {
                int idx = rangeIndex + pulseIndex * numRange;
                temp_g[idx] *= tempZero;
            }
            for(int pulseIndex = rightIdx; pulseIndex < numSamples; pulseIndex++) {
                int idx = rangeIndex + pulseIndex * numRange;
                temp_g[idx] *= tempZero;
            }
        }

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Getting G

        // For each row
        for (int rangeIndex = 0; rangeIndex < numRange; rangeIndex++) {
            CArray<__nTp> phaseData = temp_g[std::slice(rangeIndex, numSamples, numRange)];

            // ifftw(phaseData);
            // CArray<__nTp> compressed_range = fftshift(phaseData);
            CArray<__nTp> compressed_range = fftshift(phaseData);
            fftw(compressed_range);
            // CArray<__nTp> compressed_range = fftshift(phaseData);
            // ifftw(compressed_range);

            for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
                G[rangeIndex + pulseIndex * numRange] = compressed_range[pulseIndex];
                // G[rangeIndex + pulseIndex * numRange] = phaseData[pulseIndex];
            }
        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Finding G_dot
        // For each row (diff by row)
        for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
            for(int pulseNum = 0; pulseNum < numSamples - 1; pulseNum++) {
                G_dot[rangeNum + pulseNum * numRange] = G[rangeNum + (pulseNum + 1) * numRange] - G[rangeNum + pulseNum * numRange];
            }
            // G_dot[rangeNum + (numSamples - 1) * numRange] = G_dot[rangeNum + (numSamples - 2) * numRange];
        }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Finding Phi_dot and Phi
        phi_dot[0] = 0;
        for (int pulseNum = 0; pulseNum < numSamples-1; pulseNum++) {
            double G_norm = 0; // Something to temporarily hold the summed data Norm for that sample
            for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
                int idx = rangeNum + pulseNum * numRange;
                Complex<__nTp> temp = Complex<__nTp>::conj(G[idx]) * G_dot[idx];
                phi_dot[pulseNum + 1] += temp.imag();
                // phi_dot[pulseNum] += (G[idx]._M_real * G_dot[idx]._M_imag) +
                //                      (-1 * G[idx]._M_imag * G_dot[idx]._M_real); // Only the imaginary component is needed
                // G_norm += sqrt(data[idx]._M_real * data[idx]._M_real - data[idx]._M_imag * data[idx]._M_imag);
                G_norm += (Complex<__nTp>::abs(G[idx]) * Complex<__nTp>::abs(G[idx]));
            }
            // printf("idx = %d, val = %e, g_norm = %e\n", pulseNum, phi_dot[pulseNum], G_norm);
            phi_dot[pulseNum + 1] /= G_norm;
            // printf("idx = %d, val = %e\n", pulseNum, G_norm);
        }

        for (int pulseNum = 1; pulseNum < numSamples; pulseNum++) { // Integrate to get phi
            // printf("idx = %d, val = %f\n", pulseNum, phi_dot[pulseNum]);
            phi_dot[pulseNum] += phi_dot[pulseNum - 1];
            // printf("idx = %d, val = %f\n", pulseNum, phi_dot[pulseNum]);
        }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Removing the linear trend
        // Don't know if removing the linar trend is needed, will check after applying the correction
        // TODO: Try removing the linear trend
        //       Figure out what's causing the numerical instability

        double sumX = 0.0;
        double sumY = 0.0;
        double sumXY = 0.0;
        double sumXX = 0.0;

        for (int i = 0; i < numSamples; i++) {
            sumX += i;
            sumY += phi_dot[i];
            sumXY += i*phi_dot[i];
            sumXX += i * i;
        }

        double numS = numSamples * sumXY - sumX * sumY;
        double den = numSamples * sumXX - sumX * sumX;

        double numC = sumY * sumXX - sumX * sumXY;

        double temp1 = numS/den;
        double temp2 = numC/den;
        double tempa =  temp1;
        double tempb =  temp2;
        // printf("tempa = %f\ntempb = %f\na = %f\nb = %f\nD = %f\n", tempa, tempb, a, b, D);
        for(int i = 0; i < numSamples; i++) {
            phi_dot[i] -= (tempa * i + tempb);
            // printf("idx = %d, val = %f\n", i, phi_dot[i]);
        }

////////////////////////////////////////////////////////////////////////////////////////////
// Condition Check

        double rms = 0;
        for(int i = 0; i < numSamples; i++) {
            // printf("iteration = %d, idx = %d, phi_dot = %e\n", iii, i, phi_dot[i]);
            rms += (phi_dot[i] * phi_dot[i]);
        }
        rms /= numSamples;
        rms = sqrt(rms);
        printf("rms = %f\n", rms);
        if(rms < 0.1) break;

/////////////////////////////////////////////////////////////////////////////////////////////
// Applying the correction
        CArray<__nTp>temp_img(data, numSamples*numRange);

        // For each row
        double alpha = 1;
        for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
            CArray<__nTp> phaseData = temp_img[std::slice(rangeNum, numSamples, numRange)];
            fftw(phaseData);                                                 
            // ifftw(phaseData);
            // CArray<__nTp> compressed_range = fftshift(phaseData);
            for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
                Complex<__nTp> tempExp(cos(alpha * phi_dot[pulseNum]),
                                        sin(-1 * alpha * phi_dot[pulseNum])); // Something to represent e^(-j*phi)
                // int idx = rangeNum + pulseNum * numRange;
                // compressed_range[pulseNum] *= tempExp;
                phaseData[pulseNum] *= tempExp;
            }

            // fftw(compressed_range);
            ifftw(phaseData);
            CArray<__nTp> compressed_range = phaseData.cshift((phaseData.size()));
            // ifftw(compressed_range);
            // compressed_range = fftshift(compressed_range);
            // compressed_range = fftshift(compressed_range);
            for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
                int idx = rangeNum + pulseNum * numRange;
                data[idx] = compressed_range[pulseNum];
                // data[idx] = phaseData[rangeNum];
            }
        }

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Done
    }

    printf("Stopping iteration: %d\n", iii);
    delete[] G;
    delete[] G_dot;
    delete[] phi_dot;
}

template<typename __nTp, typename __nTpParams>
void run_pga(const SAR_Aperture<__nTp> &sar_data,
            const SAR_ImageFormationParameters<__nTpParams> &sar_image_params,
            CArray<__nTp> &output_image) {

    std::cout << "Running backprojection SAR focusing algorithm." << std::endl;
    L_(linfo) << "Running backprojection SAR focusing algorithm.";
    /*
        % Calculate the range to every bin in the range profile (m)
        data.r_vec = linspace(-data.Nfft/2,data.Nfft/2-1,data.Nfft)*data.maxWr/data.Nfft;
     */

    if (sar_image_params.zeropad_fft == true) {
        // zero-pad data to have length N = 2^ceil(log2(SIZE))) where
        // SIZE denotes the number of range samples per pulse
    }


    // Calculate range bins for range compression-based algorithms, e.g., backprojection
    RangeBinData<__nTp> range_bin_data;
    range_bin_data.rangeBins.shape.push_back(sar_image_params.N_fft);
    range_bin_data.rangeBins.shape.push_back(1);
    range_bin_data.rangeBins.data.resize(sar_image_params.N_fft);
    __nTp *rangeBins = &range_bin_data.rangeBins.data[0]; //[sar_image_params.N_fft];
    __nTp &minRange = range_bin_data.minRange;
    __nTp &maxRange = range_bin_data.maxRange;

    minRange = std::numeric_limits<float>::infinity();
    maxRange = -std::numeric_limits<float>::infinity();
    for (int rIdx = 0; rIdx < sar_image_params.N_fft; rIdx++) {
        // -maxWr/2:maxWr/Nfft:maxWr/2
        //float rVal = ((float) rIdx / Nfft - 0.5f) * maxWr;
        __nTp rVal = RANGE_INDEX_TO_RANGE_VALUE(rIdx, sar_image_params.max_Wy_m, sar_image_params.N_fft);
        rangeBins[rIdx] = rVal;
        if (minRange > rangeBins[rIdx]) {
            minRange = rangeBins[rIdx];
        }
        if (maxRange < rangeBins[rIdx]) {
            maxRange = rangeBins[rIdx];
        }
    }

    __nTp timeleft = 0.0f;

    const Complex<__nTp> *range_profiles_cast = static_cast<const Complex<__nTp> *> (&sar_data.sampleData.data[0]);
    //mxComplexSingleClass* output_image_cast = static_cast<mxComplexSingleClass*> (output_image);

    CArray<__nTp> range_profiles_arr(range_profiles_cast, sar_data.numAzimuthSamples * sar_data.numRangeSamples);

    for (int pulseIndex = 0; pulseIndex < sar_data.numAzimuthSamples; pulseIndex++) {
        if (pulseIndex > 1 && (pulseIndex % 100) == 0) {
            L_(linfo) << "Pulse " << pulseIndex << " of " << sar_data.numAzimuthSamples
                      << ", " << std::setprecision(2) << timeleft << " minutes remaining";
            std::cout << "Pulse " << pulseIndex << " of " << sar_data.numAzimuthSamples
                      << ", " << std::setprecision(2) << timeleft << " minutes remaining" << std::endl;
        }

        CArray<__nTp> phaseData = range_profiles_arr[std::slice(pulseIndex * sar_image_params.N_fft,
                                                                sar_image_params.N_fft, 1)];

        //ifft(phaseData);
        ifftw(phaseData);

        CArray<__nTp> compressed_range = fftshift(phaseData);
        computeDifferentialRangeAndPhaseCorrections(pulseIndex, sar_data,
                                                    sar_image_params, compressed_range, range_bin_data,
                                                    output_image);
    }

    // Degrade image with random 10th order polynomail phase
    int order = 10;
    double coeffs[order] = {0};
    for(int i = 0; i < order; i++) {
        // coeffs[i] = (std::rand()/RAND_MAX - 0.5) * sar_image_params.N_x_pix;
        coeffs[i] = (std::rand()/RAND_MAX - 0.5) * 500;
    }
    double ph_err[sar_image_params.N_x_pix] = {0};
    double start = -1;
    double end = 1;
    double spacing = (end - start) / (sar_image_params.N_x_pix - 1);
    for(int xIdx = 0; xIdx < sar_image_params.N_x_pix; xIdx++) {
        for(int coeffIdx = 0; coeffIdx < order; coeffIdx++) {
            ph_err[xIdx] += pow(coeffs[coeffIdx], order - coeffIdx - 1) * (start + spacing * xIdx);
        }
    }

    double sumX = 0.0;
    double sumY = 0.0;
    double sumXY = 0.0;
    double sumXX = 0.0;

    for (int i = 0; i < sar_image_params.N_x_pix; i++) {
        sumX += i;
        sumY += ph_err[i];
        sumXY += i*ph_err[i];
        sumXX += i * i;
    }

    double numS = sar_image_params.N_x_pix * sumXY - sumX * sumY;
    double den = sar_image_params.N_x_pix * sumXX - sumX * sumX;

    double numC = sumY * sumXX - sumX * sumXY;

    double temp1 = numS/den;
    double temp2 = numC/den;
    double tempa =  temp1;
    double tempb =  temp2;

    for(int i = 0; i < sar_image_params.N_x_pix; i++) {
        ph_err[i] -= (tempa * i + tempb);
        // printf("idx = %d, val = %f\n", i, phi_dot[i]);
    }

    // For each row
    // random number between 0 - 1/100
    double ph2_err[sar_image_params.N_x_pix] = {0};
    for (int xIdx = 0; xIdx < sar_image_params.N_x_pix; xIdx++) {
        ph2_err[xIdx] = 50 * PI / 180 * ((double) std::rand())/RAND_MAX;
    }
    for (int yIdx = 0; yIdx < sar_image_params.N_y_pix; yIdx++) {
        // float eps = 2*PI*((double) std::rand())/RAND_MAX;
        // float eps = 2 * PI * ((double)xIdx) / sar_image_params.N_x_pix;
        // Complex<__nTp> tempComplex(cos(eps), sin(eps));
        CArray<__nTp> phaseData = output_image[std::slice(yIdx, sar_image_params.N_x_pix, sar_image_params.N_y_pix)];

        ifftw(phaseData);

        CArray<__nTp> compressed_range = fftshift(phaseData);
        for (int xIdx = 0; xIdx < sar_image_params.N_x_pix; xIdx++) {
            // float eps = 100 * PI * ((double)xIdx) / sar_image_params.N_x_pix;
            // float eps = PI / 4;
            // Complex<__nTp> tempComplex(cos(eps), sin(eps));
            Complex<__nTp> tempComplex(cos(ph_err[xIdx]), sin(ph_err[xIdx]));
            compressed_range[xIdx] *= tempComplex;
        }

        CArray<__nTp> compressed_range2 = fftshift(compressed_range);
        fftw(compressed_range2);
        for (int xIdx = 0; xIdx < sar_image_params.N_x_pix; xIdx++) {
            int idx = yIdx + xIdx * sar_image_params.N_y_pix;

            output_image[idx] = compressed_range2[xIdx];
        }
    }

    Complex<__nTp> temp_out[sar_image_params.N_x_pix * sar_image_params.N_y_pix];
    for(int iii = 0; iii < sar_image_params.N_x_pix * sar_image_params.N_y_pix; iii++)
        temp_out[iii] = output_image[iii];
    autofocus<__nTp>(temp_out, sar_image_params.N_x_pix, sar_image_params.N_y_pix, 30);
    Complex<__nTp> tempDiff(0,0);
    float floatDiff = 0;
    for(int iii = 0; iii < sar_image_params.N_x_pix * sar_image_params.N_y_pix; iii++) {
        tempDiff += output_image[iii] - temp_out[iii];
        floatDiff += abs(Complex<__nTp>::abs(output_image[iii]) - Complex<__nTp>::abs(temp_out[iii]));
    }
    std::cout << std::endl << "Output difference after autofocus: " << tempDiff << std::endl;
    std::cout << std::endl << "Magnitude difference after autofocus: " << floatDiff << std::endl;
    for(int iii = 0; iii < sar_image_params.N_x_pix * sar_image_params.N_y_pix; iii++)
        output_image[iii] = temp_out[iii];
}

template<typename __nTp, typename __nTpParams>
void autofocus_SAR_image(const SAR_Aperture<__nTp> &sar_data,
                         const SAR_ImageFormationParameters<__nTpParams> &sar_image_params,
                         CArray<__nTp> &output_image) {
    // Display maximum scene size and resolution
    std::cout << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << sar_image_params.max_Wy_m << " m range, "
              << sar_image_params.max_Wx_m << " m cross-range" << std::endl;
    L_(linfo) << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << sar_image_params.max_Wy_m << " m range, "
              << sar_image_params.max_Wx_m << " m cross-range";
    std::cout << "Resolution:  " << std::fixed << std::setprecision(2) << sar_image_params.slant_rangeResolution
              << "m range, " << sar_image_params.azimuthResolution << " m cross-range" << std::endl;
    L_(linfo) << "Resolution:  " << std::fixed << std::setprecision(2) << sar_image_params.slant_rangeResolution
              << "m range, "  << sar_image_params.azimuthResolution << " m cross-range";

    switch (sar_image_params.algorithm) {
        case SAR_ImageFormationParameters<__nTpParams>::ALGORITHM::PHASE_GRADIENT_ALGORITHM:
            std::cout << "Selected phase gradient algorithm for focusing." << std::endl;
            if (sar_image_params.N_fft != sar_data.numRangeSamples) {
                std::cout << "Zero padding SAR samples." << std::endl;
                // TODO:
                // Zero pad the SAR samples in the range direction
                // Recalculate the aperture frequency samples
            }
            run_pga(sar_data, sar_image_params, output_image);
            break;
        case SAR_ImageFormationParameters<__nTpParams>::ALGORITHM::MATCHED_FILTER:
            std::cout << "Selected matched filtering algorithm for focusing." << std::endl;
            run_mf(sar_data, sar_image_params, output_image);
            break;
        default:
            std::cout << "focus_SAR_image()::Algorithm requested is not recognized or available." << std::endl;
    }

}

void cxxopts_integration_local(cxxopts::Options &options) {

    options.add_options()
            ("i,input", "Input file", cxxopts::value<std::string>())
            ("k,pulseSkip", "Number of pulses to skip for estimation", cxxopts::value<int>()->default_value("1"))
            ("m,multi", "Multiresolution Value", cxxopts::value<int>()->default_value("1"))
            ("n,numPulse", "Number of pulses to focus", cxxopts::value<int>()->default_value("0"))
            ("s,style", "Linear or Quadratic Calculation", cxxopts::value<int>()->default_value("0"))
            //("f,format", "Data format {GOTCHA, Sandia, <auto>}", cxxopts::value<std::string>()->default_value("auto"))
            ("p,polarity", "Polarity {HH,HV,VH,VV,<any>}", cxxopts::value<std::string>()->default_value("any"))
            ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
            ("v,verbose", "Enable verbose output", cxxopts::value<bool>(verbose))
            ("r,dynrange", "Dynamic Range (dB) <70 dB>", cxxopts::value<float>()->default_value("70"))
            ("o,output", "Output file <sar_image.bmp>", cxxopts::value<std::string>()->default_value("sar_image.bmp"))
            ("h,help", "Print usage");
}

int main(int argc, char **argv) {
    ComplexType test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    ComplexType out[8];
    ComplexArrayType data(test, 8);
    std::unordered_map<std::string, matvar_t *> matlab_readvar_map;

    cxxopts::Options options("cpuBackProjection",
                             "UNC Charlotte Machine Vision Lab SAR Back Projection focusing code.");
    cxxopts_integration_local(options);

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    bool debug = result["debug"].as<bool>();
    int multiRes = result["multi"].as<int>();
    int style = result["style"].as<int>();
    int nPulse = result["numPulse"].as<int>();
    int pulseSkip = result["pulseSkip"].as<int>();

    initialize_Sandia_SPHRead(matlab_readvar_map);
    initialize_GOTCHA_MATRead(matlab_readvar_map);

    std::string inputfile;
    if (result.count("input")) {
        inputfile = result["input"].as<std::string>();
    } else {
        std::stringstream ss;

        // Sandia SAR DATA FILE LOADING
        int file_idx = 9; // 1-10 for Sandia Rio Grande, 1-9 for Sandia Farms
        std::string fileprefix = Sandia_RioGrande_fileprefix;
        std::string filepostfix = Sandia_RioGrande_filepostfix;
        //        std::string fileprefix = Sandia_Farms_fileprefix;
        //        std::string filepostfix = Sandia_Farms_filepostfix;
        ss << std::setfill('0') << std::setw(2) << file_idx;


        // GOTCHA SAR DATA FILE LOADING
        int azimuth = 1; // 1-360 for all GOTCHA polarities=(HH,VV,HV,VH) and pass=[pass1,...,pass7] 
        //        std::string fileprefix = GOTCHA_fileprefix;
        //        std::string filepostfix = GOTCHA_filepostfix;
        //        ss << std::setfill('0') << std::setw(3) << azimuth;

        inputfile = fileprefix + ss.str() + filepostfix + ".mat";
    }

    std::cout << "Successfully opened MATLAB file " << inputfile << "." << std::endl;

    SAR_Aperture<NumericType> SAR_aperture_data;
    if (read_MAT_Variables(inputfile, matlab_readvar_map, SAR_aperture_data) == EXIT_FAILURE) {
        std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
        return EXIT_FAILURE;
    }
    // Print out raw data imported from file
    std::cout << SAR_aperture_data << std::endl;

    // Sandia SAR data is multi-channel having up to 4 polarities
    // 1 = HH, 2 = HV, 3 = VH, 4 = VVbandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
    std::string polarity = result["polarity"].as<std::string>();
    if ((polarity == "HH" || polarity == "any") && SAR_aperture_data.sampleData.shape.size() >= 1) {
        SAR_aperture_data.polarity_channel = 0;
    } else if (polarity == "HV" && SAR_aperture_data.sampleData.shape.size() >= 2) {
        SAR_aperture_data.polarity_channel = 1;
    } else if (polarity == "VH" && SAR_aperture_data.sampleData.shape.size() >= 3) {
        SAR_aperture_data.polarity_channel = 2;
    } else if (polarity == "VV" && SAR_aperture_data.sampleData.shape.size() >= 4) {
        SAR_aperture_data.polarity_channel = 3;
    } else {
        std::cout << "Requested polarity channel " << polarity << " is not available." << std::endl;
        return EXIT_FAILURE;
    }
    if (SAR_aperture_data.sampleData.shape.size() > 2) {
        SAR_aperture_data.format_GOTCHA = false;
        // the dimensional index of the polarity index in the 
        // multi-dimensional array (for Sandia SPH SAR data)
        SAR_aperture_data.polarity_dimension = 2;
    }

    initialize_SAR_Aperture_Data(SAR_aperture_data);

    SAR_ImageFormationParameters<NumericType> SAR_image_params =
            SAR_ImageFormationParameters<NumericType>();

    // to increase the frequency samples to a power of 2
    // SAR_image_params.N_fft = (int) 0x01 << (int) (ceil(log2(SAR_aperture_data.numRangeSamples)));
    SAR_image_params.N_fft = (int) SAR_aperture_data.numRangeSamples;
    //SAR_image_params.N_fft = aperture.numRangeSamples;
    SAR_image_params.N_x_pix = (int) SAR_aperture_data.numAzimuthSamples;
    //SAR_image_params.N_y_pix = image_params.N_fft;
    SAR_image_params.N_y_pix = (int) SAR_aperture_data.numRangeSamples;
    // focus image on target phase center
    // Determine the maximum scene size of the image (m)
    // max down-range/fast-time/y-axis extent of image (m)
    SAR_image_params.max_Wy_m = CLIGHT / (2.0 * SAR_aperture_data.mean_deltaF);
    // max cross-range/fast-time/x-axis extent of image (m)
    SAR_image_params.max_Wx_m =
            CLIGHT / (2.0 * std::abs(SAR_aperture_data.mean_Ant_deltaAz) * SAR_aperture_data.mean_startF);

    // default view is 100% of the maximum possible view
    SAR_image_params.Wx_m = 1.00 * SAR_image_params.max_Wx_m;
    SAR_image_params.Wy_m = 1.00 * SAR_image_params.max_Wy_m;
    // make reconstructed image equal size in (x,y) dimensions
    SAR_image_params.N_x_pix = (int) ((float) SAR_image_params.Wx_m * SAR_image_params.N_y_pix) / SAR_image_params.Wy_m;
    // Determine the resolution of the image (m)
    SAR_image_params.slant_rangeResolution = CLIGHT / (2.0 * SAR_aperture_data.mean_bandwidth);
    SAR_image_params.ground_rangeResolution =
            SAR_image_params.slant_rangeResolution / std::sin(SAR_aperture_data.mean_Ant_El);
    SAR_image_params.azimuthResolution = CLIGHT / (2.0 * SAR_aperture_data.Ant_totalAz * SAR_aperture_data.mean_startF);

    // Print out data after critical data fields for SAR focusing have been computed
    std::cout << SAR_aperture_data << std::endl;

    SAR_Aperture<NumericType> SAR_focusing_data;
    if (!SAR_aperture_data.format_GOTCHA) {
        //SAR_aperture_data.exportData(SAR_focusing_data, SAR_aperture_data.polarity_channel);
        SAR_aperture_data.exportData(SAR_focusing_data, 2);
    } else {
        SAR_focusing_data = SAR_aperture_data;
    }

    //    SAR_ImageFormationParameters<NumericType> SAR_image_params =
    //            SAR_ImageFormationParameters<NumericType>::create<NumericType>(SAR_focusing_data);

    if (nPulse > 2) {
        SAR_focusing_data.numAzimuthSamples = nPulse;
    }

    std::cout << "Data for focusing" << std::endl;
    std::cout << SAR_focusing_data << std::endl;

    std::ofstream myfile;
    myfile.open("collectedData.txt", std::ios::out | std::ios::app);
    myfile << inputfile.c_str() << ',';

    printf("Main: deltaAz = %f, deltaF = %f, mean_startF = %f\nmaxWx_m = %f, maxWy_m = %f, Wx_m = %f, Wy_m = %f\nX_pix = %d, Y_pix = %d\nNum Az = %d, Num range = %d\n",
           SAR_aperture_data.mean_Ant_deltaAz, SAR_aperture_data.mean_startF, SAR_aperture_data.mean_deltaF,
           SAR_image_params.max_Wx_m, SAR_image_params.max_Wy_m, SAR_image_params.Wx_m, SAR_image_params.Wy_m,
           SAR_image_params.N_x_pix, SAR_image_params.N_y_pix, SAR_aperture_data.numAzimuthSamples,
           SAR_aperture_data.numRangeSamples);
    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);

    if (multiRes < 1) multiRes = 1;
    SAR_image_params.algorithm = SAR_ImageFormationParameters<NumericType>::ALGORITHM::PHASE_GRADIENT_ALGORITHM;
    autofocus_SAR_image(SAR_focusing_data, SAR_image_params, output_image);

    // Required parameters for output generation manually overridden by command line arguments
    std::string output_filename = result["output"].as<std::string>();
    SAR_image_params.dyn_range_dB = result["dynrange"].as<float>();

    writeBMPFile(SAR_image_params, output_image, output_filename);
    myfile << '\n';
    myfile.close();
    return EXIT_SUCCESS;
}
