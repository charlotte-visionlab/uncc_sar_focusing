/* 
 * File:   gpuBackProjectionKernel.cuh
 * Author: arwillis
 *
 * Created on June 12, 2021, 10:29 AM
 */

#ifndef CPUBACKPROJECTION_HPP
#define CPUBACKPROJECTION_HPP

#include "uncc_sar_focusing.hpp"
#include "cpuBackProjection_fft.hpp"

template <typename __nTp>
Complex<__nTp> interp1(const __nTp* xSampleLocations, const int nSamples, const CArray<__nTp>& sampleValues, const float xInterpLocation, const __nTp xIndex) {
    Complex<__nTp> iVal(0, 0);
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
    iVal = alpha * sampleValues[rightIdx] + (__nTp(1.0) - alpha) * sampleValues[rightIdx - 1];
    return iVal;
}

// idx should be integer    
#define RANGE_INDEX_TO_RANGE_VALUE(idx, maxWr, N) ((float) idx / N - 0.5f) * maxWr
// val should be float
#define RANGE_VALUE_TO_RANGE_INDEX(val, maxWr, N) (val / maxWr + 0.5f) * N

template <typename __nTp, typename __nTpParams>
void computeDifferentialRangeAndPhaseCorrections(int pulseIndex,
        const SAR_Aperture<__nTp>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        const CArray<__nTp>& rangeCompressed, 
        const RangeBinData<__nTp> range_bin_data,
        CArray<__nTp>& output_image) {

    float target_x, target_y, target_z;
    target_z = 0;
    float delta_x = SARImgParams.Wx_m / (SARImgParams.N_x_pix - 1);
    float delta_y = SARImgParams.Wy_m / (SARImgParams.N_y_pix - 1);
    //std::cout << "(minRvec,maxRvec) = (" << min_Rvec << ", " << max_Rvec << ")" << std::endl;
    target_x = SARImgParams.x0_m - (SARImgParams.Wx_m / 2);
    
    const __nTp* r_vec = &range_bin_data.rangeBins.data[0];
    const __nTp& min_Rvec = range_bin_data.minRange;
    const __nTp& max_Rvec = range_bin_data.maxRange;
    for (int xIdx = 0; xIdx < SARImgParams.N_x_pix; xIdx++) {
        target_y = SARImgParams.y0_m - (SARImgParams.Wy_m / 2);
        for (int yIdx = 0; yIdx < SARImgParams.N_y_pix; yIdx++) {
            float dR_val = std::sqrt((SARData.Ant_x.data[pulseIndex] - target_x) * (SARData.Ant_x.data[pulseIndex] - target_x) +
                    (SARData.Ant_y.data[pulseIndex] - target_y) * (SARData.Ant_y.data[pulseIndex] - target_y) +
                    (SARData.Ant_z.data[pulseIndex] - target_z) * (SARData.Ant_z.data[pulseIndex] - target_z)) - SARData.slant_range.data[pulseIndex];
            //  std::cout << "y= " << target.y << " dR(" << xIdx << ", " << yIdx << ") = " << dR_val << std::endl;
            if (dR_val > min_Rvec && dR_val < max_Rvec) {
                // TODO: Ambiguate as default double and have it cast to float if appropriate for precision specifications
                Complex<__nTp> phCorr_val = Complex<__nTp>::polar(1.0f, (__nTp) ((4.0 * PI * SARData.startF.data[pulseIndex] * dR_val) / CLIGHT));
                //Complex<__nTp> phCorr_val = std::polar(__nTp(1.0), (__nTp) ((4.0 * PI * SARData.startF.data[pulseIndex] * dR_val) / CLIGHT));
                //std::cout << "idx = " << (xIdx * Ny_pix + yIdx) << " (x,y)=(" << target.x << "," << target.y << ")"
                //        << "(dR,phCorr)=(" << dR_val << ", " << phCorr_val << ")" << std::endl;
                __nTp dR_idx = RANGE_VALUE_TO_RANGE_INDEX(dR_val, SARImgParams.max_Wy_m, SARImgParams.N_fft);
                Complex<__nTp> iRC_val = interp1(r_vec, SARImgParams.N_fft, rangeCompressed, dR_val, dR_idx);
                //int outputIdx = xIdx * SARImgParams.N_y_pix + yIdx;
                //std::cout << "output[" << outputIdx << "] += " << (iRC_val * phCorr_val) << std::endl;
                output_image[xIdx * SARImgParams.N_y_pix + yIdx] += iRC_val * phCorr_val;
            }
            target_y += delta_y;
        }
        target_x += delta_x;
    }
}

template <typename __nTp, typename __nTpParams>
void run_bp(const SAR_Aperture<__nTp>& sar_data,
        const SAR_ImageFormationParameters<__nTpParams>& sar_image_params,
        CArray<__nTp>& output_image) {

    std::cout << "Running backprojection SAR focusing algorithm." << std::endl;
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
    __nTp* rangeBins = &range_bin_data.rangeBins.data[0]; //[sar_image_params.N_fft];
    __nTp& minRange = range_bin_data.minRange;
    __nTp& maxRange = range_bin_data.maxRange;

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

    const Complex<__nTp>* range_profiles_cast = static_cast<const Complex<__nTp>*> (&sar_data.sampleData.data[0]);
    //mxComplexSingleClass* output_image_cast = static_cast<mxComplexSingleClass*> (output_image);

    CArray<__nTp> range_profiles_arr(range_profiles_cast, sar_data.numAzimuthSamples * sar_data.numRangeSamples);

    for (int pulseIndex = 0; pulseIndex < sar_data.numAzimuthSamples; pulseIndex++) {
        if (pulseIndex > 1 && (pulseIndex % 100) == 0) {
            std::cout << "Pulse " << pulseIndex << " of " << sar_data.numAzimuthSamples
                    << ", " << std::setprecision(2) << timeleft << " minutes remaining" << std::endl;
        }

        CArray<__nTp> phaseData = range_profiles_arr[std::slice(pulseIndex * sar_image_params.N_fft, sar_image_params.N_fft, 1)];

        //ifft(phaseData);
        ifftw(phaseData);

        CArray<__nTp> compressed_range = fftshift(phaseData);
        computeDifferentialRangeAndPhaseCorrections(pulseIndex, sar_data,
                sar_image_params, compressed_range, range_bin_data,
                output_image);
    }
}

template <typename __nTp, typename __nTpParams>
void computeDifferentialRangeAndPhaseCorrectionsMF(int pulseIndex,
        const SAR_Aperture<__nTp>& sar_data,
        const SAR_ImageFormationParameters<__nTpParams>& sar_image_params,
        CArray<__nTp>& output_image) {

    float target_x, target_y, target_z;
    target_z = 0;
    float delta_x = sar_image_params.Wx_m / (sar_image_params.N_x_pix - 1);
    float delta_y = sar_image_params.Wy_m / (sar_image_params.N_y_pix - 1);
    //std::cout << "(minRvec,maxRvec) = (" << min_Rvec << ", " << max_Rvec << ")" << std::endl;
    target_x = sar_image_params.x0_m - (sar_image_params.Wx_m / 2);
    for (int xIdx = 0; xIdx < sar_image_params.N_x_pix; xIdx++) {
        target_y = sar_image_params.y0_m - (sar_image_params.Wy_m / 2);
        for (int yIdx = 0; yIdx < sar_image_params.N_y_pix; yIdx++) {
            float dR_val = std::sqrt((sar_data.Ant_x.data[pulseIndex] - target_x) * (sar_data.Ant_x.data[pulseIndex] - target_x) +
                    (sar_data.Ant_y.data[pulseIndex] - target_y) * (sar_data.Ant_y.data[pulseIndex] - target_y) +
                    (sar_data.Ant_z.data[pulseIndex] - target_z) * (sar_data.Ant_z.data[pulseIndex] - target_z)) - sar_data.slant_range.data[pulseIndex];
            //  std::cout << "y= " << target.y << " dR(" << xIdx << ", " << yIdx << ") = " << dR_val << std::endl;
            //int outputIdx = xIdx * SARImgParams.N_y_pix + yIdx;
            //std::cout << "output[" << outputIdx << "] += " << (iRC_val * phCorr_val) << std::endl;
            int pulse_startF_FreqIdx = pulseIndex * sar_data.numRangeSamples;
            for (int freqIdx = 0; freqIdx < sar_data.numRangeSamples; freqIdx++) {
                // TODO: Amiguate as default double and have it cast to float if appropriate for precision specifications
                const Complex<__nTp>& phaseHistorySample = sar_data.sampleData.data[pulse_startF_FreqIdx + freqIdx];
                //Complex<__nTp> phCorr_val = std::polar(1.0f, (__nTp) ((4.0 * PI * SARData.freq.data[pulse_startF_FreqIdx + freqIdx] * dR_val) / CLIGHT));
                Complex<__nTp> phCorr_val = Complex<__nTp>::polar(1.0f, (__nTp) ((4.0 * PI * sar_data.freq.data[pulse_startF_FreqIdx + freqIdx] * dR_val) / CLIGHT));
                output_image[xIdx * sar_image_params.N_y_pix + yIdx] += phaseHistorySample * phCorr_val;
            }
            target_y += delta_y;
        }
        target_x += delta_x;
    }
}

template <typename __nTp, typename __nTpParams>
void run_mf(const SAR_Aperture<__nTp>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        CArray<__nTp>& output_image) {

    std::cout << "Running matched filter SAR focusing algorithm." << std::endl;
    __nTp timeleft = 0.0f;

    const Complex<__nTp>* range_profiles_cast = static_cast<const Complex<__nTp>*> (&SARData.sampleData.data[0]);
    //mxComplexSingleClass* output_image_cast = static_cast<mxComplexSingleClass*> (output_image);

    CArray<__nTp> range_profiles_arr(range_profiles_cast, SARData.numAzimuthSamples * SARData.numRangeSamples);

    for (int pulseIndex = 0; pulseIndex < SARData.numAzimuthSamples; pulseIndex++) {
        if (pulseIndex > 1) {// && (pulseIndex % 100) == 0) {
            std::cout << "Pulse " << pulseIndex << " of " << SARData.numAzimuthSamples
                    << ", " << std::setprecision(2) << timeleft << " minutes remaining" << std::endl;

            computeDifferentialRangeAndPhaseCorrectionsMF(pulseIndex, SARData,
                    SARImgParams, output_image);
        }
    }
}

template <typename __nTp, typename __nTpParams>
void focus_SAR_image(const SAR_Aperture<__nTp>& SARData,
        const SAR_ImageFormationParameters<__nTpParams>& SARImgParams,
        CArray<__nTp>& output_image) {

    // Display maximum scene size and resolution
    std::cout << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << SARImgParams.max_Wy_m << " m range, "
            << SARImgParams.max_Wx_m << " m cross-range" << std::endl;
    std::cout << "Resolution:  " << std::fixed << std::setprecision(2) << SARImgParams.slant_rangeResolution << "m range, "
            << SARImgParams.azimuthResolution << " m cross-range" << std::endl;

    run_bp(SARData, SARImgParams, output_image);
    //run_mf(SARData, SARImgParams, output_image);
}

#endif /* CPUBACKPROJECTION_HPP */

