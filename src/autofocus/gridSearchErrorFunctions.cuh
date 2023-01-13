#ifndef GRIDSEARCH_ERROR_FUNCTIONS
#define GRIDSEARCH_ERROR_FUNCTIONS

#include <iostream>
#include <cufft.h>
#include <uncc_sar_focusing.hpp>

#define CUDAFUNCTION __host__ __device__

template<typename __Tp>
CUDAFUNCTION void grid_backprojection_loop(const cufftComplex* sampleData,
        const int numRangeSamples, const int numAzimuthSamples,
        const __Tp delta_x_m_per_pix, const __Tp delta_y_m_per_pix,
        const __Tp left, const __Tp bottom,
        const __Tp rmin, const __Tp rmax,
        const __Tp* Ant_x,
        const __Tp* Ant_y,
        const __Tp* Ant_z,
        const __Tp* slant_range,
        const __Tp* startF,
        const SAR_ImageFormationParameters<__Tp>* sar_image_params,
        const __Tp* range_vec,
        cufftComplex* output_image) {

    int x_pix = (blockIdx.x * blockDim.x) + threadIdx.x;

    for(int y_pix = 0; y_pix < sar_image_params->N_y_pix; y_pix++){
        if (x_pix >= sar_image_params->N_x_pix || y_pix >= sar_image_params->N_y_pix) {
            return;
        }
        float xpos_m = left + x_pix * delta_x_m_per_pix;
        float ypos_m = bottom + y_pix * delta_y_m_per_pix;
        Complex<float> xy_pix_SLC_return(0, 0);

        for (int pulseNum = 0; pulseNum < numAzimuthSamples; ++pulseNum) {
            float R = sqrtf(
                    (xpos_m - Ant_x[pulseNum]) * (xpos_m - Ant_x[pulseNum]) +
                    (ypos_m - Ant_y[pulseNum]) * (ypos_m - Ant_y[pulseNum]) +
                    Ant_z[pulseNum] * Ant_z[pulseNum]);

            float dR_val = R - slant_range[pulseNum];

            if (dR_val > rmin && dR_val < rmax) {
                Complex<float> phCorr_val = Complex<float>::polar(1.0f, (float) ((4.0 * PI * startF[pulseNum] * dR_val) / CLIGHT));
                float dR_idx = (dR_val / sar_image_params->max_Wy_m + 0.5f) * sar_image_params->N_fft;
                int rightIdx = (int) roundf(dR_idx);

                float alpha = (dR_val - range_vec[rightIdx - 1]) / (range_vec[rightIdx] - range_vec[rightIdx - 1]);
                Complex<float> lVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].x, sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].y);
                Complex<float> rVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx].x, sampleData[pulseNum * sar_image_params->N_fft + rightIdx].y);
                Complex<float> iRC_val = alpha * rVal + (float(1.0) - alpha) * lVal;
                xy_pix_SLC_return += iRC_val * phCorr_val;
            }
        }
        output_image[y_pix].x = xy_pix_SLC_return.real();
        output_image[y_pix].y = xy_pix_SLC_return.imag();
    }
}

template<typename __Tp>
CUDAFUNCTION void temp_grid_backprojection_loop(const cufftComplex* sampleData,
        const int numRangeSamples, const int numAzimuthSamples,
        const __Tp delta_x_m_per_pix, const __Tp delta_y_m_per_pix,
        const __Tp left, const __Tp bottom,
        const __Tp rmin, const __Tp rmax,
        const __Tp* Ant_x,
        const __Tp* Ant_y,
        const __Tp* Ant_z,
        const __Tp* slant_range,
        const __Tp* startF,
        const SAR_ImageFormationParameters<__Tp>* sar_image_params,
        const __Tp* range_vec,
        cufftComplex* output_image) {

    int x_pix = (blockIdx.x * blockDim.x) + threadIdx.x;

    for(int y_pix = 0; y_pix < sar_image_params->N_y_pix; y_pix++){
        if (x_pix >= sar_image_params->N_x_pix || y_pix >= sar_image_params->N_y_pix) {
            return;
        }
        float xpos_m = left + x_pix * delta_x_m_per_pix;
        float ypos_m = bottom + y_pix * delta_y_m_per_pix;
        Complex<float> xy_pix_SLC_return(0, 0);

        for (int pulseNum = 0; pulseNum < numAzimuthSamples; ++pulseNum) {
            float R = sqrtf(
                    (xpos_m - Ant_x[pulseNum]) * (xpos_m - Ant_x[pulseNum]) +
                    (ypos_m - Ant_y[pulseNum]) * (ypos_m - Ant_y[pulseNum]) +
                    Ant_z[pulseNum] * Ant_z[pulseNum]);

            float dR_val = R - slant_range[pulseNum];

            if (dR_val > rmin && dR_val < rmax) {
                Complex<float> phCorr_val = Complex<float>::polar(1.0f, (float) ((4.0 * PI * startF[pulseNum] * dR_val) / CLIGHT));
                float dR_idx = (dR_val / sar_image_params->max_Wy_m + 0.5f) * sar_image_params->N_fft;
                int rightIdx = (int) roundf(dR_idx);
                float alpha = (dR_val - range_vec[rightIdx - 1]) / (range_vec[rightIdx] - range_vec[rightIdx - 1]);
                Complex<float> lVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].x, sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].y);
                Complex<float> rVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx].x, sampleData[pulseNum * sar_image_params->N_fft + rightIdx].y);
                Complex<float> iRC_val = alpha * rVal + (float(1.0) - alpha) * lVal;
                xy_pix_SLC_return += iRC_val * phCorr_val;
            }
        }
        output_image[(x_pix * sar_image_params->N_y_pix) + y_pix].x = xy_pix_SLC_return.real();
        output_image[(x_pix * sar_image_params->N_y_pix) + y_pix].y = xy_pix_SLC_return.imag();
    }
}

CUDAFUNCTION void cuCalculateHistogram(unsigned int* hist, unsigned char * image, int width, int height) {
    for (int x = 0; x < height; x++) {
        if (x >= height) return;
        hist[image[x]] += 1;
    }
}

template<typename __nTp>
CUDAFUNCTION __nTp getAbs(cufftComplex a) {
    __nTp __x = a.x;
    __nTp __y = a.y;
    __nTp __s = std::abs(__x) > std::abs(__y) ? std::abs(__x) : std::abs(__y);
    if (__s == 0.0f) return 0.0f;
    __x /= __s;
    __y /= __s;
    return __s * std::sqrt(__x * __x + __y * __y);
}

template<typename __nTp>
__device__ float cuCalculateColumnEntropy(const SAR_ImageFormationParameters<__nTp>* SARImgParams, cufftComplex* output_image) {

    unsigned int width = SARImgParams->N_x_pix, height = SARImgParams->N_y_pix;
    
    unsigned char pixels[424] = {0};
    __shared__ __nTp max_val;
    unsigned int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ __nTp tempTempVal[451];
    __shared__ __nTp tempTempEntropy[451];
    if(threadIdx.x == 0) {
        max_val = 0;
        for(int i = 0; i < 451; i++) {
            tempTempVal[i] = 0;
            tempTempEntropy[i] = 0;
        }
    }
    __syncthreads();
    for(int i = 0; i < height; i++) {
        __nTp tempVal = getAbs<__nTp>(output_image[i]);
        // if (max_val < tempVal) max_val = tempVal; 
        if(tempTempVal[xIdx] < tempVal) tempTempVal[xIdx] = tempVal;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        for(int i = 0; i < 451; i++) 
            if(max_val < tempTempVal[i]) max_val = tempTempVal[i];
    }
    __syncthreads();

    bool flipY = false;
    bool flipX = true;
    int srcIndex;
    for (int y_dstIndex = 0; y_dstIndex < SARImgParams->N_y_pix; y_dstIndex++) {
        cufftComplex SARpixel = output_image[y_dstIndex];
        float pixelf = (float) (255.0 / SARImgParams->dyn_range_dB)*
                ((20 * std::log10(getAbs<__nTp>(SARpixel) / max_val)) + SARImgParams->dyn_range_dB);
        unsigned char pixel = (pixelf < 0) ? 0 : (unsigned char) pixelf;
        pixels[y_dstIndex] = pixel;
    }

    __nTp tempEntropy = 0;
    unsigned int hist[256] = {0};
    for (int x = 0; x < height; x++) {
        hist[(unsigned int)pixels[x]] += 1;
    }
    __nTp prob = 0.0;
    for (int y_dstIndex = 0; y_dstIndex < height; y_dstIndex++) {
        prob = (float)hist[(unsigned int)pixels[y_dstIndex]]/(float)height;
        tempTempEntropy[xIdx] -= (prob*log2(prob));
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        for(int i = 0; i < 451; i++) {
            tempEntropy += tempTempEntropy[i];
        }
    }
    __syncthreads();
    return tempEntropy/width;
}

template<typename func_precision, typename grid_precision, uint32_t D, typename __Tp>
__device__ func_precision kernelWrapper(nv_ext::Vec<grid_precision, D> &parameters,
    cufftComplex* sampleData,
    int numRangeSamples, int numAzimuthSamples,
    __Tp delta_x_m_per_pix, __Tp delta_y_m_per_pix,
    __Tp left, __Tp bottom,
    __Tp rmin, __Tp rmax,
    __Tp* Ant_x,
    __Tp* Ant_y,
    __Tp* Ant_z,
    __Tp* slant_range,
    __Tp* startF,
    SAR_ImageFormationParameters<__Tp>* sar_image_params,
    __Tp* range_vec) {

    unsigned int width = sar_image_params->N_x_pix, height = sar_image_params->N_y_pix;
    float output = 0;

    cufftComplex output_image[424];
    __Tp Ant_x_new[119] = {0};
    __Tp Ant_y_new[119] = {0};
    __Tp Ant_z_new[119] = {0};

    // Ant_x_new[0] = Ant_x[0];
    // Ant_y_new[0] = Ant_y[0];
    // Ant_z_new[0] = Ant_z[0];

    // for(int i = 0; i < numAzimuthSamples-1; i++) {
    //     Ant_x_new[1+i] = Ant_x_new[i]+parameters[0]*i*i+parameters[1]*i+parameters[2];
    //     Ant_y_new[1+i] = Ant_y_new[i]+parameters[3]*i*i+parameters[4]*i+parameters[5];
    //     Ant_z_new[1+i] = Ant_z_new[i]+parameters[6]*i*i+parameters[7]*i+parameters[8];
    // }

    for(int i = 0; i < numAzimuthSamples; i++) {
        Ant_x_new[i] = parameters[0] * i * i + parameters[1] * i + parameters[2];
        Ant_y_new[i] = parameters[3] * i * i + parameters[4] * i + parameters[5];
        Ant_z_new[i] = parameters[6] * i * i + parameters[7] * i + parameters[8];
    }

    grid_backprojection_loop(sampleData, 
                        numRangeSamples, numAzimuthSamples, 
                        delta_x_m_per_pix, delta_y_m_per_pix,
                        left, bottom,
                        rmin, rmax,
                        Ant_x_new, Ant_y_new, Ant_z_new,
                        slant_range,
                        startF,
                        sar_image_params,
                        range_vec,
                        output_image);

    output = cuCalculateColumnEntropy(sar_image_params, output_image);
    __syncthreads();

    if(threadIdx.x == 0){
        // For some reason if it's not printed out it returns 0
        printf("%f\n",output);
        return output;
    } 
    else
        return 0;
}

template<typename func_precision, typename grid_precision, uint32_t D, typename __Tp>
__global__ void computeImageKernel(nv_ext::Vec<grid_precision, D> parameters,
    cufftComplex* sampleData,
    int numRangeSamples, int numAzimuthSamples,
    __Tp delta_x_m_per_pix, __Tp delta_y_m_per_pix,
    __Tp left, __Tp bottom,
    __Tp rmin, __Tp rmax,
    __Tp* Ant_x,
    __Tp* Ant_y,
    __Tp* Ant_z,
    __Tp* slant_range,
    __Tp* startF,
    SAR_ImageFormationParameters<__Tp>* sar_image_params,
    __Tp* range_vec,
    cufftComplex* output_image) {

    unsigned int width = sar_image_params->N_x_pix, height = sar_image_params->N_y_pix;

    __Tp Ant_x_new[119];
    __Tp Ant_y_new[119];
    __Tp Ant_z_new[119];

    // Ant_x_new[0] = Ant_x[0];
    // Ant_y_new[0] = Ant_y[0];
    // Ant_z_new[0] = Ant_z[0];

    // for(int i = 0; i < numAzimuthSamples-1; i++) {
    //     Ant_x_new[1+i] = Ant_x_new[i]+parameters[0]*i*i+parameters[1]*i+parameters[2];
    //     Ant_y_new[1+i] = Ant_y_new[i]+parameters[3]*i*i+parameters[4]*i+parameters[5];
    //     Ant_z_new[1+i] = Ant_z_new[i]+parameters[6]*i*i+parameters[7]*i+parameters[8];
    // }

    for(int i = 0; i < numAzimuthSamples; i++) {
        Ant_x_new[i] = parameters[0] * i * i + parameters[1] * i + parameters[2];
        Ant_y_new[i] = parameters[3] * i * i + parameters[4] * i + parameters[5];
        Ant_z_new[i] = parameters[6] * i * i + parameters[7] * i + parameters[8];
    }

    temp_grid_backprojection_loop(sampleData, 
                        numRangeSamples, numAzimuthSamples, 
                        delta_x_m_per_pix, delta_y_m_per_pix,
                        left, bottom,
                        rmin, rmax,
                        Ant_x_new, Ant_y_new, Ant_z_new,
                        slant_range,
                        startF,
                        sar_image_params,
                        range_vec,
                        output_image);
}

#endif