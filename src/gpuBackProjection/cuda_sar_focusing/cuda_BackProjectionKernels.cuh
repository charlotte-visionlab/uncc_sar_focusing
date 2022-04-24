/*
 * Copyright (C) 2021 Andrew R. Willis
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
#ifndef GPUBACKPROJECTIONKERNELS_CUH
#define GPUBACKPROJECTIONKERNELS_CUH

#include <cufft.h>

#include <cuda_runtime.h>

#include <uncc_sar_focusing.hpp>

#define CAREFUL_AMINUSB_SQ(x,y) __fmul_rn(__fadd_rn((x), -1.0f*(y)), __fadd_rn((x), -1.0f*(y)))

#define BLOCKWIDTH    16
#define BLOCKHEIGHT   16

#define MAKERADIUS(xpixel, ypixel, xa, ya, za) sqrtf(CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + __fmul_rn(za, za))

template<typename __Tp>
__global__ void backprojection_loop(const cufftComplex* sampleData,
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
    int y_pix = (blockIdx.y * blockDim.y) + threadIdx.y;
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
        //float R = MAKERADIUS(xpos_m, ypos_m, Ant_x[pulseNum], Ant_y[pulseNum], Ant_z[pulseNum]);

        float dR_val = R - slant_range[pulseNum];

        if (dR_val > rmin && dR_val < rmax) {
            Complex<float> phCorr_val = Complex<float>::polar(1.0f, (float) ((4.0 * PI * startF[pulseNum] * dR_val) / CLIGHT));
            float dR_idx = (dR_val / sar_image_params->max_Wy_m + 0.5f) * sar_image_params->N_fft;
            int rightIdx = (int) roundf(dR_idx);
            //while (++rightIdx < sar_image_params->N_fft && range_vec[rightIdx] <= dR_val);
            float alpha = (dR_val - range_vec[rightIdx - 1]) / (range_vec[rightIdx] - range_vec[rightIdx - 1]);
            Complex<float> lVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].x, sampleData[pulseNum * sar_image_params->N_fft + rightIdx - 1].y);
            Complex<float> rVal(sampleData[pulseNum * sar_image_params->N_fft + rightIdx].x, sampleData[pulseNum * sar_image_params->N_fft + rightIdx].y);
            Complex<float> iRC_val = alpha * rVal + (float(1.0) - alpha) * lVal;
            //if (abs(xpos_m) < 0.5 && abs(ypos_m) < 0.5) {
            //    printf("pulse=%d (x,y)=(%f,%f) platform(x,y,z)=(%f,%f,%f) R0=%f R=%f dR_val=%f dR_idx = %f (rmin,rmax) = (%f,%f) rightIdx = %d range_vec[ridx-1] = %f range_vec[ridx] = %f\n", pulseNum, xpos_m, ypos_m,
            //            Ant_x[pulseNum], Ant_y[pulseNum], Ant_z[pulseNum], slant_range[pulseNum], R, dR_val, dR_idx, rmin, rmax,
            //            rightIdx, range_vec[rightIdx - 1], range_vec[rightIdx]);
            //}
            xy_pix_SLC_return += iRC_val * phCorr_val;
        }
    }
    //if (x_pix == 0 || x_pix == sar_image_params->N_x_pix - 1) {
    //    printf("(x_pix,y_pix)=(%d,%d)\n", x_pix, y_pix);
    //}
    //if (abs(xpos_m) < 0.5 && abs(ypos_m) < 0.5) {
    //    printf("(Npulses,Nfft)=(%d,%d) (N_x_pix,N_y_pix)=(%d,%d) \n",
    //            numAzimuthSamples, sar_image_params->N_fft,
    //            sar_image_params->N_x_pix, sar_image_params->N_y_pix);
    //}
    output_image[(x_pix * sar_image_params->N_y_pix) + y_pix].x = xy_pix_SLC_return.real();
    output_image[(x_pix * sar_image_params->N_y_pix) + y_pix].y = xy_pix_SLC_return.imag();
}

__global__
void cufftNormalize_1DBatch_kernel(cufftComplex* data, const int N, const int num_batches) {
    int batch_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int value_index = (blockIdx.y * blockDim.y) + threadIdx.y;
    int fvalue_offset = N * batch_index + value_index;
    if (batch_index < num_batches && value_index < N) {
        data[fvalue_offset].x /= N;
        data[fvalue_offset].y /= N;        
    }
}

void cufftNormalize_1DBatch(cufftComplex *data, const int N, const int num_batches) {
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3(std::ceil((float) num_batches / 16),
            std::ceil((float) N / 16));
    cufftNormalize_1DBatch_kernel << < grid, block >>> (data, N, num_batches);
}

template <typename T >
__global__
void cufftShift_1DBatch_kernel(T* data, int N, int num_batches) {
    int batch_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int value_index = (blockIdx.y * blockDim.y) + threadIdx.y;
    int fvalue_offset = N * batch_index + value_index;
    if (batch_index < num_batches && value_index < N / 2) {
        // Save the first value
        T regTemp = data[fvalue_offset];
        // Swap the first element
        data[fvalue_offset] = (T) data[fvalue_offset + (N / 2)];
        // Swap the second one
        data[fvalue_offset + (N / 2)] = (T) regTemp;
    }
}

template <typename T>
void cufftShift_1DBatch(T* data, int N, int num_batches) {
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3(std::ceil((float) num_batches / 16),
            std::ceil((float) N / 16));
    cufftShift_1DBatch_kernel << < grid, block >>> (data, N, num_batches);
}

#endif /* GPUBACKPROJECTIONKERNELS_CUH */

