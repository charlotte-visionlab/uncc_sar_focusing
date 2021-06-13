/* 
 * File:   gpuBackProjectionKernel.cuh
 * Author: arwillis
 *
 * Created on June 12, 2021, 10:29 AM
 */

#ifndef GPUBACKPROJECTIONKERNEL_CUH
#define GPUBACKPROJECTIONKERNEL_CUH

#define mexPrintf printf

#define REAL(vec) (vec.x)
#define IMAG(vec) (vec.y)

#define CAREFUL_AMINUSB_SQ(x,y) __fmul_rn(__fadd_rn((x), -1.0f*(y)), __fadd_rn((x), -1.0f*(y)))

#define BLOCKWIDTH    16
#define BLOCKHEIGHT   16

#define MAKERADIUS(xpixel,ypixel, xa,ya,za) sqrtf(CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + __fmul_rn(za, za))

#define CLIGHT 299792458.0f /* c: speed of light, m/s */
#define PI 3.14159265359f   /* pi, accurate to 6th place in single precision */

__device__ float2 expjf(float in);
__device__ float2 expjf_div_2(float in);

/* Globals and externs */

/* Complex textures containing range profiles */
texture<float2, 2, cudaReadModeElementType> tex_projections;

/* 4-elem. textures for x, y, z, r0 */
texture<float4, 1, cudaReadModeElementType> tex_platform_info;

/* Main kernel.
 *
 * Tuning options:
 * - is it worth #defining radar parameters like start_frequency?
 *      ............  or imaging parameters like xmin/ymax?
 * - Make sure (4 pi / c) is computed at compile time!
 * - Use 24-bit integer multiplications!
 *
 * */
__global__ void backprojection_loop(float2* full_image,
        int Npulses, int Ny_pix, float delta_x_m_per_pix, float delta_y_m_per_pix,
        int PROJ_LENGTH,
        int X_OFFSET, int Y_OFFSET,
        float C__4_DELTA_FREQ, float* PI_4_F0__CLIGHT,
        float left, float bottom, float min_eff_idx, float4 * platform_info,
        float * debug_effective_idx, float * debug_2, float * x_mat, float * y_mat,
        float rmin, float rmax) {

    float2 subimage;
    subimage = make_float2(0.0f, 0.0f);
    float2 csum; // For compensated sum
    float y, t;
    csum = make_float2(0.0f, 0.0f);

    float xpos_m = left + (float) (blockIdx.x * BLOCKWIDTH + threadIdx.x) *
            delta_x_m_per_pix;
    float ypos_m = bottom + (float) (blockIdx.y * BLOCKHEIGHT + threadIdx.y) *
            delta_y_m_per_pix;

    float2 texel;

    __shared__ int pulseNum;
    __shared__ float4 platform;
    __shared__ int copyblock;

    __shared__ float delta_r;
    delta_r = rmax - rmin;
    __shared__ float Nl1_dr;
    Nl1_dr = __fdiv_rn((float) PROJ_LENGTH - 1.0f, delta_r);

    copyblock = (blockIdx.y * BLOCKHEIGHT) * Ny_pix + blockIdx.x * BLOCKWIDTH;

    /* Now, let's loop through these projections! 
     * */
#pragma unroll 3
    for (pulseNum = 0; pulseNum < Npulses; ++pulseNum) {

        //platform = tex1D(tex_platform_info, (float)proj_num + 0.5f);

        platform = platform_info[pulseNum];

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
        float R = MAKERADIUS(xpos_m, ypos_m, platform.x, platform.y, platform.z);
        if (abs(xpos_m) < 0.1 && abs(ypos_m) < 0.1) {
            //printf("Nl1_dr is: %f\n", Nl1_dr);
            printf("pulse=%d (x,y)=(%f,%f) platform(x,y,z)=(%f,%f,%f) R0=%f R=%f \n", pulseNum, xpos_m, ypos_m, 
                    platform.x, platform.y, platform.z, platform.w, R);
        }

        /* Per-pixel-projection phasor = exp(1j 4 pi/c * f_min * R). */
        //float2 pixel_scale = expjf_div_2(PI_4_F0__CLIGHT[proj_num] * R * 0.5f);
        float2 pixel_scale = expjf(PI_4_F0__CLIGHT[pulseNum] * R);

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
        texel = tex2D(tex_projections, 0.5f + (float) pulseNum, 0.5f + effective_idx);

        /* Scale "texel" by "pixel_scale".
           The RHS of these 2 lines just implement complex multiplication.
         */
        y = REAL(texel) * REAL(pixel_scale) - REAL(csum);
        t = subimage.x + y;
        csum.x = (t - subimage.x) - y;
        subimage.x = t;

        y = -1.0f * IMAG(texel) * IMAG(pixel_scale) - REAL(csum);
        t = subimage.x + y;
        csum.x = (t - subimage.x) - y;
        subimage.x = t;

        y = REAL(texel) * IMAG(pixel_scale) - IMAG(csum);
        t = subimage.y + y;
        csum.y = (t - subimage.y) - y;
        subimage.y = t;

        y = IMAG(texel) * REAL(pixel_scale) - IMAG(csum);
        t = subimage.y + y;
        csum.y = (t - subimage.y) - y;
        subimage.y = t;

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
    full_image[copyblock + (threadIdx.y) * Ny_pix + threadIdx.x] = subimage;
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

#endif /* GPUBACKPROJECTIONKERNEL_CUH */

