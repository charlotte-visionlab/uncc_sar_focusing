#ifndef GRIDSEARCH_ERROR_FUNCTIONS
#define GRIDSEARCH_ERROR_FUNCTIONS

#include <iostream>
#include <cufft.h>
#include <uncc_sar_focusing.hpp>

#define CUDAFUNCTION __host__ __device__

template<typename __Tp>
CUDAFUNCTION void grid_backprojection_loop(const cufftComplex *sampleData,
                                           const int numRangeSamples, const int numAzimuthSamples,
                                           const __Tp delta_x_m_per_pix, const __Tp delta_y_m_per_pix,
                                           const __Tp left, const __Tp bottom,
                                           const __Tp rmin, const __Tp rmax,
                                           const __Tp *Ant_x,
                                           const __Tp *Ant_y,
                                           const __Tp *Ant_z,
                                           const __Tp *slant_range,
                                           const __Tp *startF,
                                           const SAR_ImageFormationParameters<__Tp> *sar_image_params,
                                           const __Tp *range_vec,
                                           cufftComplex *output_image) {

    int x_pix = (blockIdx.x * blockDim.x) + threadIdx.x;

    for (int y_pix = 0; y_pix < sar_image_params->N_y_pix; y_pix++) {
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
                Complex<float> phCorr_val = Complex<float>::polar(1.0f,
                                                                  (float) ((4.0 * PI * startF[pulseNum] * dR_val) /
                                                                           CLIGHT));
                float dR_idx = (dR_val / sar_image_params->max_Wy_m + 0.5f) * sar_image_params->N_fft;
                int rightIdx = (int) roundf(dR_idx);

                float alpha = (dR_val - range_vec[rightIdx - 1]) / (range_vec[rightIdx] - range_vec[rightIdx - 1]);
                Complex<float> lVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].x,
                                    sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].y);
                Complex<float> rVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx].x,
                                    sampleData[pulseNum * sar_image_params->N_fft + rightIdx].y);
                Complex<float> iRC_val = alpha * rVal + (float(1.0) - alpha) * lVal;
                xy_pix_SLC_return += iRC_val * phCorr_val;
            }
        }
        output_image[y_pix].x = xy_pix_SLC_return.real();
        output_image[y_pix].y = xy_pix_SLC_return.imag();
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

template<typename __Tp>
__device__ double dftCalculation(const cufftComplex *sampleData,
                               const int numRangeSamples, const int numAzimuthSamples,
                               const __Tp delta_x_m_per_pix, const __Tp delta_y_m_per_pix,
                               const __Tp left, const __Tp bottom,
                               const __Tp rmin, const __Tp rmax,
                               const __Tp *Ant_x,
                               const __Tp *Ant_y,
                               const __Tp *Ant_z,
                               const __Tp *slant_range,
                               const __Tp *startF,
                               const SAR_ImageFormationParameters<__Tp> *sar_image_params,
                               const __Tp *range_vec,
                               cufftComplex *output_image) {

    int dft_x = (blockIdx.x * blockDim.x) + threadIdx.x;

    Complex<float>* dft_out = new Complex<float>[sar_image_params->N_y_pix];
    double temp_out = 0.0;
    __shared__ float dft_holder[500];
    __shared__ __Tp out;
    __shared__ __Tp max_val;
    
    if (dft_x == 0) {
    	out = 0;
    	max_val = 0;
        for(int idx = 0; idx < sar_image_params->N_x_pix; idx++)
            dft_holder[idx] = 0;
            
        for (int i = 0; i < sar_image_params->N_x_pix * sar_image_params->N_y_pix; i++) {
		    __Tp tempVal = getAbs<__Tp>(output_image[i]);
		    // if (max_val < tempVal) max_val = tempVal; 
		    if (max_val < tempVal) max_val = tempVal;
		}
    }
    
    __syncthreads();

    for (int dft_y = 0; dft_y < sar_image_params->N_y_pix; dft_y++) {
        for(int x_pix = 0; x_pix < sar_image_params->N_x_pix; x_pix++) {
    
		    cufftComplex SARpixel = output_image[(x_pix * sar_image_params->N_y_pix) + dft_y];
        	float pixelf = (float) (255.0 / sar_image_params->dyn_range_dB) *
            	           ((20 * std::log10(getAbs<__Tp>(SARpixel) / max_val)) + sar_image_params->dyn_range_dB);
        	unsigned char pixel = (pixelf < 0) ? 0 : (unsigned char) pixelf;
		    Complex<float> xy_pix_SLC_return(pixel, 0.0);

		    // for (int pulseNum = 0; pulseNum < numAzimuthSamples; ++pulseNum) {
		    //     float R = sqrtf(
		    //             (xpos_m - Ant_x[pulseNum]) * (xpos_m - Ant_x[pulseNum]) +
		    //             (ypos_m - Ant_y[pulseNum]) * (ypos_m - Ant_y[pulseNum]) +
		    //             Ant_z[pulseNum] * Ant_z[pulseNum]);

		    //     float dR_val = R - slant_range[pulseNum];

		    //     if (dR_val > rmin && dR_val < rmax) {
		    //         Complex<float> phCorr_val = Complex<float>::polar(1.0f,
		    //                                                           (float) ((4.0 * PI * startF[pulseNum] * dR_val) /
		    //                                                                    CLIGHT));
		    //         float dR_idx = (dR_val / sar_image_params->max_Wy_m + 0.5f) * sar_image_params->N_fft;
		    //         int rightIdx = (int) roundf(dR_idx);

		    //         float alpha = (dR_val - range_vec[rightIdx - 1]) / (range_vec[rightIdx] - range_vec[rightIdx - 1]);
		    //         Complex<float> lVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].x,
		    //                             sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].y);
		    //         Complex<float> rVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx].x,
		    //                             sampleData[pulseNum * sar_image_params->N_fft + rightIdx].y);
		    //         Complex<float> iRC_val = alpha * rVal + (float(1.0) - alpha) * lVal;
		    //         xy_pix_SLC_return += iRC_val * phCorr_val;
		    //     }
		    // }
		    
		    Complex<float> w_n(cos(-2 * M_PI * dft_x * x_pix / sar_image_params->N_x_pix), sin(-2 * M_PI * dft_x * x_pix / sar_image_params->N_x_pix));
		    Complex<float> temp(0,0);
		    
		    temp = w_n * xy_pix_SLC_return;
		    dft_out[dft_y] += temp;
		}
    }
    
    for (int dft_y = 0; dft_y < sar_image_params->N_y_pix; dft_y++) {
    	temp_out += sqrt(dft_out[dft_y].real() * dft_out[dft_y].real() + dft_out[dft_y].imag() * dft_out[dft_y].imag());
    }
    
    dft_holder[dft_x] = temp_out;
    delete[] dft_out;
    __syncthreads();
    
    if (dft_x == 0) {
    	int idxEnd = sar_image_params->N_x_pix * 0.25;
    	for (int idx = 0; idx < idxEnd; idx++) {
    		out += 20*log10(dft_holder[idx]);
    	}
    }
    
    __syncthreads();
    
    return out;
}

template<typename __Tp>
__device__ double dft2DCalculation(const cufftComplex *sampleData,
                                 const int numRangeSamples, const int numAzimuthSamples,
                                 const __Tp delta_x_m_per_pix, const __Tp delta_y_m_per_pix,
                                 const __Tp left, const __Tp bottom,
                                 const __Tp rmin, const __Tp rmax,
                                 const __Tp *Ant_x,
                                 const __Tp *Ant_y,
                                 const __Tp *Ant_z,
                                 const __Tp *slant_range,
                                 const __Tp *startF,
                                 const SAR_ImageFormationParameters<__Tp> *sar_image_params,
                                 const __Tp *range_vec,
                                 cufftComplex *output_image) {

    int dft_x = (blockIdx.x * blockDim.x) + threadIdx.x;

	const double percentage = 0.35;
    Complex<float>* dft_out1 = new Complex<float>[sar_image_params->N_y_pix];
    Complex<float>* dft_out2 = new Complex<float>[sar_image_params->N_y_pix];
    double temp_out = 0.0;
    __shared__ double dft_holder[500];
    __shared__ double out;
    __shared__ __Tp max_val;
    
    if (dft_x == 0) {
    	out = 0;
    	max_val = 0;
        for(int idx = 0; idx < sar_image_params->N_x_pix; idx++)
            dft_holder[idx] = 0;
            
        for (int i = 0; i < sar_image_params->N_x_pix * sar_image_params->N_y_pix; i++) {
		    __Tp tempVal = getAbs<__Tp>(output_image[i]);
		    // if (max_val < tempVal) max_val = tempVal; 
		    if (max_val < tempVal) max_val = tempVal;
		}
    }
    
    __syncthreads();

    for (int dft_y = 0; dft_y < sar_image_params->N_y_pix; dft_y++) {
        for(int x_pix = 0; x_pix < sar_image_params->N_x_pix; x_pix++) {
		    cufftComplex SARpixel = output_image[(x_pix * sar_image_params->N_y_pix) + dft_y];
        	float pixelf = (float) (255.0 / sar_image_params->dyn_range_dB) *
            	           ((20 * std::log10(getAbs<__Tp>(SARpixel) / max_val)) + sar_image_params->dyn_range_dB);
        	unsigned char pixel = (pixelf < 0) ? 0 : (unsigned char) pixelf;
		    Complex<float> xy_pix_SLC_return(pixel, 0.0);
		    
		    Complex<float> w_n(cos(-2 * M_PI * dft_x * x_pix / sar_image_params->N_x_pix), sin(-2 * M_PI * dft_x * x_pix / sar_image_params->N_x_pix));
		    Complex<float> temp(0,0);
		    
		    temp = w_n * xy_pix_SLC_return;
		    dft_out1[dft_y] += temp;
		}
    }
    
    for (int dft_y = 0; dft_y < sar_image_params->N_y_pix; dft_y++) {
        for(int y_pix = 0; y_pix < sar_image_params->N_y_pix; y_pix++) {
		    Complex<float> w_n(cos(-2 * M_PI * dft_y * y_pix / sar_image_params->N_y_pix), sin(-2 * M_PI * dft_y * y_pix / sar_image_params->N_y_pix));
		    Complex<float> temp(0,0);
		    
		    temp = w_n * dft_out1[y_pix];
		    dft_out2[dft_y] += temp;
		}
    }
    
    int h_height = sar_image_params->N_y_pix / 2;
    int h_width = sar_image_params->N_x_pix / 2;
    int threshold = (h_height * h_height + h_width * h_width) * (percentage * percentage);
    
    for (int y = 0; y < h_height; y++) {
	    int r = dft_x * dft_x + y * y;
	    int r2 = (sar_image_params->N_x_pix - dft_x - 1) * (sar_image_params->N_x_pix - dft_x - 1) + y * y;
	    if (r < threshold || r2 < threshold) {
		    cufftComplex dft_temp;
		    dft_temp.x = dft_out2[y].real();
		    dft_temp.y = dft_out2[y].imag();
	    	dft_holder[dft_x] += getAbs<__Tp>(dft_temp);
	    	
	    	dft_temp.x = dft_out2[sar_image_params->N_y_pix - y - 1].real();
	    	dft_temp.y = dft_out2[sar_image_params->N_y_pix - y - 1].imag();
			dft_holder[dft_x] += getAbs<__Tp>(dft_temp);
	    }
    }
    
    delete[] dft_out1;
    delete[] dft_out2;
    __syncthreads();
    
    if (dft_x == 0) {
    	for (int idx = 0; idx < sar_image_params->N_x_pix; idx++) {
    		out += dft_holder[idx];
    	}
    }
    
    __syncthreads();
    
    return out;
}

template<typename __Tp>
CUDAFUNCTION void temp_grid_backprojection_loop(const cufftComplex *sampleData,
                                                const int numRangeSamples, const int numAzimuthSamples,
                                                const __Tp delta_x_m_per_pix, const __Tp delta_y_m_per_pix,
                                                const __Tp left, const __Tp bottom,
                                                const __Tp rmin, const __Tp rmax,
                                                const __Tp *Ant_x,
                                                const __Tp *Ant_y,
                                                const __Tp *Ant_z,
                                                const __Tp *slant_range,
                                                const __Tp *startF,
                                                const SAR_ImageFormationParameters<__Tp> *sar_image_params,
                                                const __Tp *range_vec,
                                                cufftComplex *output_image) {

    int x_pix = (blockIdx.x * blockDim.x) + threadIdx.x;

    for (int y_pix = 0; y_pix < sar_image_params->N_y_pix; y_pix++) {
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
                Complex<float> phCorr_val = Complex<float>::polar(1.0f,
                                                                  (float) ((4.0 * PI * startF[pulseNum] * dR_val) /
                                                                           CLIGHT));
                float dR_idx = (dR_val / sar_image_params->max_Wy_m + 0.5f) * sar_image_params->N_fft;
                int rightIdx = (int) roundf(dR_idx);
                float alpha = (dR_val - range_vec[rightIdx - 1]) / (range_vec[rightIdx] - range_vec[rightIdx - 1]);
                Complex<float> lVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].x,
                                    sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].y);
                Complex<float> rVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx].x,
                                    sampleData[pulseNum * sar_image_params->N_fft + rightIdx].y);
                Complex<float> iRC_val = alpha * rVal + (float(1.0) - alpha) * lVal;
                xy_pix_SLC_return += iRC_val * phCorr_val;
            }
        }
        output_image[(x_pix * sar_image_params->N_y_pix) + y_pix].x = xy_pix_SLC_return.real();
        output_image[(x_pix * sar_image_params->N_y_pix) + y_pix].y = xy_pix_SLC_return.imag();
    }
}

CUDAFUNCTION void cuCalculateHistogram(unsigned int *hist, unsigned char *image, int width, int height) {
    for (int x = 0; x < height; x++) {
        if (x >= height) return;
        hist[image[x]] += 1;
    }
}

template<typename __nTp>
__device__ float
cuCalculateColumnEntropy(const SAR_ImageFormationParameters<__nTp> *SARImgParams, cufftComplex *output_image) {

    unsigned int width = SARImgParams->N_x_pix, height = SARImgParams->N_y_pix;

    unsigned char * pixels = new unsigned char[height];
    __shared__ __nTp max_val;
    unsigned int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ __nTp tempTempVal[500];
    __shared__ double tempTempEntropy[500];
    if (threadIdx.x == 0) {
        max_val = 0;
        for (int i = 0; i < width; i++) {
            tempTempVal[i] = 0;
            tempTempEntropy[i] = 0;
        }
    }
    for (int i = 0; i < height; i++) {
        pixels[i] = 0;
    }
    __syncthreads();
    for (int i = 0; i < height; i++) {
        __nTp tempVal = getAbs<__nTp>(output_image[i]);
        // if (max_val < tempVal) max_val = tempVal; 
        if (tempTempVal[xIdx] < tempVal) tempTempVal[xIdx] = tempVal;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < width; i++)
            if (max_val < tempTempVal[i]) max_val = tempTempVal[i];
    }
    __syncthreads();
//    bool flipY = false;
//    bool flipX = true;
//    int srcIndex;
    for (int y_dstIndex = 0; y_dstIndex < SARImgParams->N_y_pix; y_dstIndex++) {
        cufftComplex SARpixel = output_image[y_dstIndex];
        float pixelf = (float) (255.0 / SARImgParams->dyn_range_dB) *
                       ((20 * std::log10(getAbs<__nTp>(SARpixel) / max_val)) + SARImgParams->dyn_range_dB);
        unsigned char pixel = (pixelf < 0) ? 0 : (unsigned char) pixelf;
        pixels[y_dstIndex] = pixel;
    }

    __shared__ __nTp tempEntropy;
    if(threadIdx.x == 0) {
       tempEntropy = 0;
    }
        
    // // Total Entropy
    // __shared__ unsigned int hist[256];
    // if (threadIdx.x == 0) {
    //     for(int i = 0; i < 256; i++) {
    //         hist[i] = 0;
    //     }
    // }
    // __syncthreads();
    // for (int x = 0; x < height; x++) {
    //     atomicAdd(hist+((unsigned int)pixels[x]), 1);
    // }
    // __syncthreads();
    // delete[] pixels;
    // double prob = 0.0;

    // for (int hist_idx = 0; hist_idx < 256; hist_idx++) {
    //     if( hist[hist_idx] == 0)
    //         continue;

    //    prob = ((double) hist[hist_idx]) / ((double)(width*height));
    //     tempTempEntropy[xIdx] -= (prob * log2(prob));
    // }

    // __syncthreads();

    // return -1*tempTempEntropy[xIdx];

    // // Column Entropy
    unsigned int hist[256] = {0};
    for (int x = 0; x < height; x++) {
        hist[(unsigned int) pixels[x]] += 1;
    }
    
    __syncthreads();
            
    delete[] pixels;
    double prob = 0.0;

    for (int hist_idx = 0; hist_idx < 256; hist_idx++) {
        if( hist[hist_idx] == 0)
            continue;
        
        prob = ((double) hist[hist_idx]) / ((double)height);
        tempTempEntropy[xIdx] -= (prob * log2(prob));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < width; i++) {
            if(tempTempEntropy[i] == 0.0f) tempEntropy += 100.0f;
            else tempEntropy -= tempTempEntropy[i];
        }
    }
    __syncthreads();
    
    return tempEntropy / width;
}

template<typename func_precision, typename grid_precision, uint32_t D, typename __Tp>
__device__ func_precision kernelWrapper(nv_ext::Vec<grid_precision, D> &parameters,
                                        cufftComplex *sampleData,
                                        int numRangeSamples, int numAzimuthSamples,
                                        __Tp delta_x_m_per_pix, __Tp delta_y_m_per_pix,
                                        __Tp left, __Tp bottom,
                                        __Tp rmin, __Tp rmax,
                                        __Tp *Ant_x,
                                        __Tp *Ant_y,
                                        __Tp *Ant_z,
                                        __Tp *slant_range,
                                        __Tp *startF,
                                        SAR_ImageFormationParameters<__Tp> *sar_image_params,
                                        __Tp *range_vec) {

    unsigned int width = sar_image_params->N_x_pix, height = sar_image_params->N_y_pix;
    float output = 0;

    __shared__ func_precision totalOut;

    cufftComplex output_image[500];
    __Tp Ant_x_new[200] = {0};
    __Tp Ant_y_new[200] = {0};
    __Tp Ant_z_new[200] = {0};

    if (D == 10) {
        for (int i = 0; i < numAzimuthSamples; i++) {
            Ant_x_new[i] = parameters[2] * i * i + parameters[1] * i * parameters[9] + parameters[0];
            Ant_y_new[i] = parameters[5] * i * i + parameters[4] * i * parameters[9] + parameters[3];
            Ant_z_new[i] = parameters[8] * i * i + parameters[7] * i * parameters[9] + parameters[6];
        }
    } else if (D == 7) {
        for (int i = 0; i < numAzimuthSamples; i++) {
            Ant_x_new[i] = parameters[1] * i * parameters[6] + parameters[0];
            Ant_y_new[i] = parameters[3] * i * parameters[6] + parameters[2];
            Ant_z_new[i] = parameters[5] * i * parameters[6] + parameters[4];
        }
    }

	cufftComplex * temp_image;
	__shared__  cufftComplex * image_pointer;
	
	if (threadIdx.x == 0) {
		temp_image = new cufftComplex[width * height];
		
		image_pointer = temp_image;
	}
	
	__syncthreads();
	
	temp_image = image_pointer;
	

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
                                  temp_image);

	__syncthreads();

	double dftOut = 0.0;
	dftOut = dft2DCalculation(sampleData,
                            numRangeSamples, numAzimuthSamples,
                            delta_x_m_per_pix, delta_y_m_per_pix,
                            left, bottom,
                            rmin, rmax,
                            Ant_x_new, Ant_y_new, Ant_z_new,
                            slant_range,
                            startF,
                            sar_image_params,
                            range_vec, 
                            temp_image);
    if (threadIdx.x == 0)
	    delete[] temp_image;
	    
    return -dftOut;

	// if (dftOut < 14540.0)
	// 	return 0;

    // grid_backprojection_loop(sampleData,
    //                          numRangeSamples, numAzimuthSamples,
    //                          delta_x_m_per_pix, delta_y_m_per_pix,
    //                          left, bottom,
    //                          rmin, rmax,
    //                          Ant_x_new, Ant_y_new, Ant_z_new,
    //                          slant_range,
    //                          startF,
    //                          sar_image_params,
    //                          range_vec,
    //                          output_image);
	
    // output = cuCalculateColumnEntropy(sar_image_params, output_image);
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     totalOut = output;
    // }
    // __syncthreads();
    // return totalOut;
    
    
    // if (threadIdx.x == 0) {
    //     // For some reason if it's not printed out it returns 0
    //     // printf("%f\n", output);
    //     return output;
    // } else
    //     return 0;
}

template<typename func_precision, typename grid_precision, uint32_t D, typename __Tp>
__global__ void computeImageKernel(nv_ext::Vec<grid_precision, D> parameters,
                                   cufftComplex *sampleData,
                                   int numRangeSamples, int numAzimuthSamples,
                                   __Tp delta_x_m_per_pix, __Tp delta_y_m_per_pix,
                                   __Tp left, __Tp bottom,
                                   __Tp rmin, __Tp rmax,
                                   __Tp *Ant_x,
                                   __Tp *Ant_y,
                                   __Tp *Ant_z,
                                   __Tp *slant_range,
                                   __Tp *startF,
                                   SAR_ImageFormationParameters<__Tp> *sar_image_params,
                                   __Tp *range_vec,
                                   cufftComplex *output_image) {

    unsigned int width = sar_image_params->N_x_pix, height = sar_image_params->N_y_pix;

    __Tp * Ant_x_new = new __Tp[numAzimuthSamples];
    __Tp * Ant_y_new = new __Tp[numAzimuthSamples];
    __Tp * Ant_z_new = new __Tp[numAzimuthSamples];

    if (D == 10) {
        // Keep the velocity between like 0.8 - 1.2
        for (int i = 0; i < numAzimuthSamples; i++) {
            Ant_x_new[i] = parameters[2] * i * i + parameters[1] * i * parameters[9] + parameters[0];
            Ant_y_new[i] = parameters[5] * i * i + parameters[4] * i * parameters[9] + parameters[3];
            Ant_z_new[i] = parameters[8] * i * i + parameters[7] * i * parameters[9] + parameters[6];
        }
    } else if (D == 7) {
        for (int i = 0; i < numAzimuthSamples; i++) {
            Ant_x_new[i] = parameters[1] * i * parameters[6] + parameters[0];
            Ant_y_new[i] = parameters[3] * i * parameters[6] + parameters[2];
            Ant_z_new[i] = parameters[5] * i * parameters[6] + parameters[4];
        }
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

    delete[] Ant_x_new;
    delete[] Ant_y_new;
    delete[] Ant_z_new;
}

#endif
