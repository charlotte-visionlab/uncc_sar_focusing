#ifndef GRIDSEARCH_ERROR_FUNCTIONS_DFT
#define GRIDSEARCH_ERROR_FUNCTIONS_DFT

#include <iostream>
#include <cufft.h>
#include <uncc_sar_focusing.hpp>

#define CUDAFUNCTION __host__ __device__

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
__device__ void grid_backprojection_loop(const cufftComplex *sampleData,
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
                                                unsigned char *output_image) {

    int x_pix = threadIdx.x;
    cufftComplex temp_output[500];
    float thread_max = 0;

    __shared__ float max_val;
    __shared__ float column_max[500];

    if (x_pix == 0) {
        max_val = 0;
        for (int idx = 0; idx < 500; idx++) {
            column_max[idx] = 0;
        }
    }

    __syncthreads();

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
        temp_output[y_pix].x = xy_pix_SLC_return.real();
        temp_output[y_pix].y = xy_pix_SLC_return.imag();

        float tempVal = getAbs<float>(temp_output[y_pix]);
        if (tempVal > thread_max) thread_max = tempVal;
    }

    column_max[x_pix] = thread_max;

    __syncthreads();

    if (x_pix == 0) {
        for (int idx = 0; idx < sar_image_params->N_x_pix; idx++) {
            if (max_val < column_max[idx]) max_val = column_max[idx];
        }
    }

    __syncthreads();

    for (int y_pix = 0; y_pix < sar_image_params->N_y_pix; y_pix++) {
        cufftComplex SARpixel = temp_output[y_pix];
        float pixelf = (float) (255.0 / sar_image_params->dyn_range_dB) *
                        ((20 * std::log10(getAbs<__Tp>(SARpixel) / max_val)) + sar_image_params->dyn_range_dB);
        unsigned char pixel = (pixelf < 0) ? 0 : (unsigned char) pixelf;

        output_image[(x_pix * sar_image_params->N_y_pix) + y_pix] = pixel;
    }
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
                                 unsigned char *output_image) {

    int dft_x = threadIdx.x;

	const double percentage = 0.35;
    Complex<float> dft_out1[500];
    Complex<float> dft_out2[500];
    double temp_out = 0.0;
    __shared__ double dft_holder[500];
    __shared__ double out;
    
    if (dft_x == 0) {
    	out = 0;
        for(int idx = 0; idx < sar_image_params->N_x_pix; idx++)
            dft_holder[idx] = 0;
    }
    
    __syncthreads();

    for (int dft_y = 0; dft_y < sar_image_params->N_y_pix; dft_y++) {
        for(int x_pix = 0; x_pix < sar_image_params->N_x_pix; x_pix++) {
            unsigned char pixel = output_image[(x_pix * sar_image_params->N_y_pix) + dft_y];
		    Complex<float> xy_pix_SLC_return(pixel, 0.0);

		    float sin_val;
            float cos_val;
            sincospif(-2.0f * dft_x * x_pix / sar_image_params->N_x_pix, &sin_val, &cos_val);
		    Complex<float> w_n(cos_val, sin_val);
		    Complex<float> temp(0,0);
		    
		    temp = w_n * xy_pix_SLC_return;
		    dft_out1[dft_y] += temp;
		}
    }
    
    for (int dft_y = 0; dft_y < sar_image_params->N_y_pix; dft_y++) {
        for(int y_pix = 0; y_pix < sar_image_params->N_y_pix; y_pix++) {
		    float sin_val;
            float cos_val;
            sincospif(-2.0f * dft_y * y_pix / sar_image_params->N_y_pix, &sin_val, &cos_val);
		    Complex<float> w_n(cos_val, sin_val);
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

    int x_pix = threadIdx.x;

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

	unsigned char * temp_image;
	__shared__  unsigned char * image_pointer;
	
	if (threadIdx.x == 0) {
		temp_image = new unsigned char[width * height];
		
		image_pointer = temp_image;
	}
	
	__syncthreads();
	
	temp_image = image_pointer;

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
