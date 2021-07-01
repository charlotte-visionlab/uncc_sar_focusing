/* 
 * File:   gpuBackProjectionKernels.cuh
 * Author: arwillis
 *
 * Created on June 12, 2021, 10:29 AM
 */

#ifndef GPUBACKPROJECTIONKERNELS_CUH
#define GPUBACKPROJECTIONKERNELS_CUH

#include <cuda_runtime.h>

#include "../../cpuBackProjection/uncc_sar_focusing.hpp"

#define REAL(vec) (vec.x)
#define IMAG(vec) (vec.y)

#define CAREFUL_AMINUSB_SQ(x,y) __fmul_rn(__fadd_rn((x), -1.0f*(y)), __fadd_rn((x), -1.0f*(y)))

#define BLOCKWIDTH    16
#define BLOCKHEIGHT   16

#define MAKERADIUS(xpixel, ypixel, xa, ya, za) sqrtf(CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + __fmul_rn(za, za))

//#define CLIGHT 299792458.0f /* c: speed of light, m/s */
//#define PI 3.14159265359f   /* pi, accurate to 6th place in single precision */

__device__ float2 expjf(float in);
__device__ float2 expjf_div_2(float in);

template<typename __Tp>
__global__ void backprojection_loop(float2* sampleData, 
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
        float2* output_image) {

    float2 subimage;
    subimage = make_float2(0.0f, 0.0f);
//    float2 csum; // For compensated sum
//    float y, t;
//    csum = make_float2(0.0f, 0.0f);
    float xpos_m = left + (float) (blockIdx.x * BLOCKWIDTH + threadIdx.x) *
            delta_x_m_per_pix;
    float ypos_m = bottom + (float) (blockIdx.y * BLOCKHEIGHT + threadIdx.y) *
            delta_y_m_per_pix;

//    float2 texel;

    __shared__ int pulseNum;
    __shared__ float4 platform;
    __shared__ int copyblock;

    __shared__ float delta_r;
    delta_r = rmax - rmin;
    __shared__ float Nl1_dr;
    Nl1_dr = __fdiv_rn((float) numRangeSamples - 1.0f, delta_r);

    copyblock = (blockIdx.y * BLOCKHEIGHT) * sar_image_params->N_y_pix + blockIdx.x * BLOCKWIDTH;

    /* Now, let's loop through these projections! 
     * */
#pragma unroll 3
    for (pulseNum = 0; pulseNum < numAzimuthSamples; ++pulseNum) {

        //platform = tex1D(tex_platform_info, (float)proj_num + 0.5f);

        //platform = platform_info[pulseNum];
        
        /* R_reciprocal = 1/R = 1/sqrt(sum_{# in xyz} [#pixel - #platform]^2),
         * This is the distance between the platform and every pixel.
         */
        /*
       float R = sqrtf( 
               (xpixel - platform.x) * 
               (xpixel - platform.x) +
               (ypixel - platform.y) * 
               (ypixel - platform.y) +
               platform.z * platform.z);*/
        float R = MAKERADIUS(xpos_m, ypos_m, Ant_x[pulseNum], Ant_y[pulseNum], Ant_z[pulseNum]);
        if (abs(xpos_m) < 0.5 && abs(ypos_m) < 0.5) {
            //printf("Nl1_dr is: %f\n", Nl1_dr);
            printf("pulse=%d (x,y)=(%f,%f) platform(x,y,z)=(%f,%f,%f) R0=%f R=%f \n", pulseNum, xpos_m, ypos_m, 
                    Ant_x[pulseNum], Ant_y[pulseNum], Ant_z[pulseNum], slant_range[pulseNum], R);
        }

        /* Per-pixel-projection phasor = exp(1j 4 pi/c * f_min * R). */
        //float2 pixel_scale = expjf_div_2(PI_4_F0__CLIGHT[proj_num] * R * 0.5f);
        //float2 pixel_scale = expjf(PI_4_F0__CLIGHT[pulseNum] * R);

        /* The fractional range bin for this pixel, this projection */
        /*
        float effective_idx = ((float)PROJ_LENGTH-1.0f) *
            (R - ( platform.w - R_START_PRE )) / (2.0f*C__4_DELTA_FREQ) 
            - min_eff_idx;*/
        //float effective_idx = ((float)PROJ_LENGTH-1.0f) / (rmax - rmin) * (R - platform.w - rmin);
        float effective_idx = __fmul_rn(Nl1_dr, __fadd_rn(__fadd_rn(R, -1.0f * platform.w), -1.0f * rmin));

        /* This is the interpolated range profile element for this pulse */

        // Flipped textures
        /*texel = tex2D(tex_projections, 
                0.5f+effective_idx, 0.5f+(float)proj_num);*/
        // offset textures
        //texel = tex2D(tex_projections, 0.5f + (float) pulseNum, 0.5f + effective_idx);

        /* Scale "texel" by "pixel_scale".
           The RHS of these 2 lines just implement complex multiplication.
         */
//        y = REAL(texel) * REAL(pixel_scale) - REAL(csum);
//        t = subimage.x + y;
//        csum.x = (t - subimage.x) - y;
//        subimage.x = t;
//
//        y = -1.0f * IMAG(texel) * IMAG(pixel_scale) - REAL(csum);
//        t = subimage.x + y;
//        csum.x = (t - subimage.x) - y;
//        subimage.x = t;
//
//        y = REAL(texel) * IMAG(pixel_scale) - IMAG(csum);
//        t = subimage.y + y;
//        csum.y = (t - subimage.y) - y;
//        subimage.y = t;
//
//        y = IMAG(texel) * REAL(pixel_scale) - IMAG(csum);
//        t = subimage.y + y;
//        csum.y = (t - subimage.y) - y;
//        subimage.y = t;

        /*
        subimage.x += REAL(texel)*REAL(pixel_scale) - 
                IMAG(texel)*IMAG(pixel_scale);
        subimage.y += REAL(texel)*IMAG(pixel_scale) + 
                IMAG(texel)*REAL(pixel_scale);
         */

        //        if (proj_num == 0) {
        //            debug_effective_idx[copyblock + (threadIdx.y) * nyout + threadIdx.x] = effective_idx;
        //            debug_2[copyblock + (threadIdx.y) * nyout + threadIdx.x] = R;
        //            x_mat[copyblock + (threadIdx.y) * nyout + threadIdx.x] = platform.x;
        //            y_mat[copyblock + (threadIdx.y) * nyout + threadIdx.x] = platform.y;
        //        }
    }
    /* Copy this thread's pixel back to global memory */
    //full_image[(blockIdx.y * BLOCKHEIGHT + threadIdx.y) * nyout + 
    //    blockIdx.x * BLOCKWIDTH + threadIdx.x] = subimage;
    output_image[copyblock + (threadIdx.y) * sar_image_params->N_y_pix + threadIdx.x] = subimage;
}

/* Credits: from BackProjectionKernal.c: "originally by reinke".
 * Given a float X, returns float2 Y = exp(j * X).
 *
 * __device__ code is always inlined. */
__device__
float2 expjf(float in) {
    float2 out;
    float t, tb;
#if USE_FAST_MATH
    t = __tanf(in / 2.0f);
#else
    t = tan(in / 2.0f);
#endif
    tb = t * t + 1.0f;
    out.x = (2.0f - tb) / tb; /* Real */
    out.y = (2.0f * t) / tb; /* Imag */

    return out;
}

__device__
float2 expjf_div_2(float in) {
    float2 out;
    float t, tb;
    //t = __tanf(in - (float)((int)(in/(PI2)))*PI2 );
    t = __tanf(in - PI * rintf(in / PI));
    tb = t * t + 1.0f;
    out.x = (2.0f - tb) / tb; /* Real */
    out.y = (2.0f * t) / tb; /* Imag */
    return out;
}

/*********************************************************************
 * Copyright © 2011-2014,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This library (cufftShift) is free software; you can redistribute it
 * and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 ********************************************************************/
 
#ifndef CUFFTSHIFT_1D_SINGLE_ARRAY_CU
#define CUFFTSHIFT_1D_SINGLE_ARRAY_CU

#include <algorithm> // std::min

#include <cuda.h>
//#include <cutil_inline.h>

unsigned int ulog2 (unsigned int u)
{
    unsigned int s, t;

    t = (u > 0xffff) << 4; u >>= t;
    s = (u > 0xff  ) << 3; u >>= s, t |= s;
    s = (u > 0xf   ) << 2; u >>= s, t |= s;
    s = (u > 0x3   ) << 1; u >>= s, t |= s;

    return (t | (u >> 1));
}

#define MIN(a,b) ((a) < (b) ? (a) : (b))

template <typename T>
void cufftShift_1D(T* data, int NX)
{
    int threadsPerBlock_X = (NX > 1) ? MIN(ulog2((unsigned int) NX), 1024) : 1;   
    dim3 grid = dim3((NX / threadsPerBlock_X), 1, 1);
    dim3 block = dim3(threadsPerBlock_X, 1, 1);;
    cufftShift_1D_kernel <<< grid, block >>> (data, NX);
}

template <typename T>
__global__
void cufftShift_1D_kernel(T* data, int NX)
{
    int threadIdxX = threadIdx.x;
    int blockDimX = blockDim.x;
    int blockIdxX = blockIdx.x;

    int index = ((blockIdxX * blockDimX) + threadIdxX);
    if (index < NX/2)
    {
        // Save the first value
        T regTemp = data[index];

        // Swap the first element
        data[index] = (T) data[index + (NX / 2)];

        // Swap the second one
        data[index + (NX / 2)] = (T) regTemp;
    }
}
#endif // CUFFTSHIFT_1D_SINGLE_ARRAY_CU

#endif /* GPUBACKPROJECTIONKERNELS_CUH */

