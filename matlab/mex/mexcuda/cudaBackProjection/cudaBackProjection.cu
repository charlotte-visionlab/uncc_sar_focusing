// Standard Library includes
#include <stdio.h>  /* printf */
#include <time.h>

// MATLAB includes
#include <mex.h>    /* Matlab junk */
#include <gpu/mxGPUArray.h>

// CUDA includes
#include <helper_cuda.h>

#include "gpuBackProjectionKernel.cuh"

#define PI2 6.2831853071800f   /* 2*pi */
#define PI_4__CLIGHT (4.0f * PI / CLIGHT)

#define ASSUME_Z_0    1     /* Ignore consult_DEM() and assume height = 0. */
#define USE_FAST_MATH 0     /* Use __math() functions? */
#define USE_RSQRT     0

#define MEXDEBUG      1

#define FLOAT_CLASS   mxSINGLE_CLASS

#ifndef VERBOSE
#define VERBOSE       0
#endif

#define ZEROCOPY      0

/***
 * Compiler logics
 * **/
#define MY_CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        printf( "Cuda error in file '%s' in line %i : %s.\n",                \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    } }

#define MY_CUDA_SAFE_CALL( call)     MY_CUDA_SAFE_CALL_NO_SYNC(call);             \

/* Pound defines from my PyCUDA implementation:
 * 
 * ---Physical constants---
 * CLIGHT
 * PI
 *
 * ---Radar/data-specific constants---
 * Delta-frequency
 * Number of projections
 *
 * ---Application runtime constants---
 * Nfft, projection length
 * Image dimensions in pixels
 * Top-left image corner
 * X/Y pixel spacing
 *
 * ---Complicated constants---
 * PI_4_F0__CLIGHT = 4*pi/clight * radar_start_frequency
 * C__4_DELTA_FREQ = clight / (4 * radar_delta_frequency)
 * R_START_PRE = C__4_DELTA_FREQ * Nfft / (Nfft-1)
 *
 * ---CUDA constants---
 * Block dimensions
 */

/***
 * Type defs
 * ***/
typedef float FloatType; /* FIXME: this should be used everywhere */

/* From ATK imager */
typedef struct {
    float * real;
    float * imag;
} complex_split;

/* To work seamlessly with Hartley's codebase */
typedef complex_split bp_complex_split;


/***
 * Prototypes
 * ***/
int gpuGetMaxGflopsDeviceId();

float2 * format_complex_to_columns(bp_complex_split a, int width_orig,
        int height_orig);

float2 * format_complex(bp_complex_split a, int size);

float4 * format_x_y_z_r(float * x, float * y, float * z, float * r, int size);

void run_bp(mxComplexSingle* phd,
        float * xObs, float * yObs, float * zObs, float * r,
        int my_num_phi, int my_proj_length, int nxout, int nyout,
        int image_chunk_width, int image_chunk_height,
        int device,
        mxComplexSingle* host_output_image,
        int start_output_index, int num_output_rows,
        float c__4_delta_freq, float pi_4_f0__clight,
        float * start_frequencies,
        float left, float right, float bottom, float top,
        float min_eff_idx, float total_proj_length);


void convert_f0(float* vec, int N) {
    int i;
    for (i = 0; i < N; ++i)
        vec[i] *= PI_4__CLIGHT;
}

float extract_f0(float* vec, int N) {
    /* Mean ...
    int i;
    float sum = 0;
    for (i=0; i<N; ++i) {
        sum += vec[i];
    }
    return sum / N;
     */
    return vec[0];
}

/* 
 * Application parameters:
 *  - range profiles
 *
 * 
 * ATK imager gets the following:
 * - range profiles (complex)
 * - f0, vector of start frequencies, Hz
 * - r0, vector of distances from radar to center of illuminated scene, m
 * - x, y, z, vectors of radar position (x points east, y north, z up), m
 * - Nimgx, Nimgy, number of pixels in x and y
 * - deltaf, spacing of frequency vector, Hz
 * - Left, right, top, bottom, corners of the square on the ground to image
 */
void mexFunction(int nlhs, /* number of LHS (output) arguments */
        mxArray* plhs[], /* array of mxArray pointers to outputs */
        int nrhs, /* number of RHS (input) args */
        const mxArray* prhs[]) /* array of pointers to inputs*/ {
    /* Section 1. 
     * These are the variables we'll use */
    /* Subsection A: these come from Matlab and are the same as the ATK code */
    mxComplexSingle* range_profiles;
    float* start_frequencies;
    float* aimpoint_ranges;
    float* xobs;
    float* yobs;
    float* zobs;
    int Nx_pix, Ny_pix;
    float delta_frequency;
    float left, right, top, bottom;

    float min_eff_idx;//, Nrangebins;

    /* Subsection B: these are computed from the matlab inputs */
    int Npulses, Nrangebins;
    float c__4_delta_freq;
    float pi_4_f0__clight;

    /* Subsection C: these are CUDA-specific options */
    int deviceId, blockwidth, blockheight;

    /* Subsection D: these are output variables */
    mxComplexSingle* host_output_image;

    // Initialize the MathWorks GPU API
    mxInitGPU();

    /* Section 2. 
     * Parse Matlab's inputs */
    range_profiles = mxGetComplexSingles(prhs[0]); 
    //range_profiles.real = (float*) mxGetPr(prhs[0]);
    //range_profiles.imag = (float*) mxGetPi(prhs[0]);

    start_frequencies = (float*) mxGetPr(prhs[1]);
    aimpoint_ranges = (float*) mxGetPr(prhs[2]);
    xobs = (float*) mxGetPr(prhs[3]);
    yobs = (float*) mxGetPr(prhs[4]);
    zobs = (float*) mxGetPr(prhs[5]);

    Nx_pix = (int) mxGetScalar(prhs[6]);
    Ny_pix = (int) mxGetScalar(prhs[7]);
    delta_frequency = (float) mxGetScalar(prhs[8]);

    left = (float) mxGetScalar(prhs[ 9]);
    right = (float) mxGetScalar(prhs[10]);
    bottom = (float) mxGetScalar(prhs[11]);
    top = (float) mxGetScalar(prhs[12]);

    /* Section 3.
     * Set up some intermediate values */

    /* Range profile dimensions */
    Npulses = mxGetN(prhs[0]);
    Nrangebins = mxGetM(prhs[0]);

    if (nrhs == 15) {
        min_eff_idx = (float) mxGetScalar(prhs[13]);
        Nrangebins = (float) mxGetScalar(prhs[14]);
    } else {
        min_eff_idx = 0;
        //Nrangebins = Nrangebins;
    }

    /* CUDA parameters
     * FIXME: these should only be preset if Matlab didn't specify them */

    // This will pick the best possible CUDA capable device
    //    devID = findCudaDevice(argc, (const char **)argv); 
    deviceId = -1;
    if (deviceId == -1) {
        // Otherwise pick the device with highest Gflops/s
        deviceId = gpuGetMaxGflopsDeviceId();
        MY_CUDA_SAFE_CALL(cudaSetDevice(deviceId));
        int major = 0, minor = 0;
        MY_CUDA_SAFE_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId));
        MY_CUDA_SAFE_CALL(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceId));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                deviceId, _ConvertSMVer2ArchName(major, minor), major, minor);
    }
    blockwidth = BLOCKWIDTH;
    blockheight = BLOCKHEIGHT;
    if (MEXDEBUG) {
        printf("WARNING: CUDA parameters not provided. Auto-selecting:\n"
                "device      %d\n"
                "blockwidth  %d\n"
                "blockheight %d\n", deviceId, blockwidth, blockheight);
    }
    
    /* Various collection-specific constants */

    c__4_delta_freq = CLIGHT / (4.0f * delta_frequency);

    /* FIXME: this TOTALLY prevents variable start frequency!!!! */
    pi_4_f0__clight = PI * 4.0f * extract_f0(start_frequencies, Npulses) / CLIGHT;
    convert_f0(start_frequencies, Npulses);

    /* Section 4.
     * Set up Matlab outputs */
    plhs[0] = mxCreateNumericMatrix(Ny_pix, Nx_pix,
            FLOAT_CLASS, mxCOMPLEX);
    host_output_image = mxGetComplexSingles(plhs[0]);
    //host_output_image.real = (float*) mxGetPr(plhs[0]);
    //host_output_image.imag = (float*) mxGetPi(plhs[0]);

    /* Section 5.
     * Call Hartley's GPU initialization & invocation code */
    run_bp(range_profiles, xobs, yobs, zobs,
            aimpoint_ranges,
            Npulses, Nrangebins, Nx_pix, Ny_pix,
            blockwidth, blockheight,
            deviceId,
            host_output_image,
            0, Ny_pix,
            c__4_delta_freq, pi_4_f0__clight,
            start_frequencies, left, right, bottom, top, min_eff_idx, Nrangebins);

    return;
}

void from_gpu_complex_to_bp_complex_split(float2 * data, mxComplexSingle* out, int size) {
    int i;
    for (i = 0; i < size; i++) {
        out[i].real = data[i].x;
        out[i].imag = data[i].y;
    }
}

float2* format_complex_to_columns(mxComplexSingle* a, int width_orig, int height_orig) {
    float2* out = (float2*) malloc(width_orig * height_orig * sizeof (float2));
    int i, j;
    for (i = 0; i < height_orig; i++) {
        int origOffset = i * width_orig;
        for (j = 0; j < width_orig; j++) {
            int newOffset = j * height_orig;
            out[newOffset + i].x = a[origOffset + j].real;
            out[newOffset + i].y = a[origOffset + j].imag;
        }
    }
    return out;
}

float2* format_complex(bp_complex_split a, int size) {
    float2* out = (float2*) malloc(size * sizeof (float2));
    int i;
    for (i = 0; i < size; i++) {
        out[i].x = a.real[i];
        out[i].y = a.imag[i];
    }
    return out;
}

float4* format_x_y_z_r(float * x, float * y, float * z, float * r, int size) {
    float4* out = (float4*) malloc(size * sizeof (float4));
    int i;
    for (i = 0; i < size; i++) {
        out[i].x = x[i];
        out[i].y = y[i];
        out[i].z = z[i];
        out[i].w = r[i];
    }
    return out;
}

void run_bp(mxComplexSingle* phd, float* xObs, float* yObs, float* zObs, float* r,
        int Npulses, int Nrangebins, int Nx_pix, int Ny_pix, int blockwidth,
        int blockheight, int deviceId, mxComplexSingle* host_output_image,
        int start_output_index, int num_output_rows,
        float c__4_delta_freq, float pi_4_f0__clight, float* start_frequencies,
        float left, float right, float bottom, float top,
        float min_eff_idx, float total_proj_length) {

    cudaDeviceProp props;

    //Get GPU information
    //    MY_CUDA_SAFE_CALL(cudaGetDevice(&deviceId));
    MY_CUDA_SAFE_CALL(cudaSetDevice(deviceId));
    MY_CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, deviceId));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
            deviceId, props.name, props.major, props.minor);

#if ZEROCOPY
    // We will want ZEROCOPY code for Xavier and newer architecture platforms
    // https://developer.ridgerun.com/wiki/index.php?title=NVIDIA_CUDA_Memory_Management
    MY_CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif
    // We will want UNIFIED MEMORY code for Maxwell architecture platforms

    int num_out_bytes = 2 * sizeof (float) * num_output_rows * Ny_pix;
    float2* out_image;


    // Set up platform data texture
    float4* trans_tex_platform_info = format_x_y_z_r(xObs, yObs, zObs, r, Npulses);
    cudaChannelFormatDesc float4desc = cudaCreateChannelDesc<float4>();
    float4* device_tex_platform_info;
    cudaArray* array_tex_platform_info;

    MY_CUDA_SAFE_CALL(cudaMallocArray(&array_tex_platform_info, &float4desc, Npulses, 1));
    MY_CUDA_SAFE_CALL(cudaMemcpyToArray(array_tex_platform_info, 0, 0,
            trans_tex_platform_info, Npulses * 4 * sizeof (float),
            cudaMemcpyHostToDevice));
    MY_CUDA_SAFE_CALL(cudaMalloc(&device_tex_platform_info, sizeof (float4) * Npulses));
    MY_CUDA_SAFE_CALL(cudaMemcpy(device_tex_platform_info, trans_tex_platform_info, Npulses * sizeof (float4),
            cudaMemcpyHostToDevice));
    //    MY_CUDA_SAFE_CALL(cudaMemcpy2DToArray(array_tex_platform_info, 0, 0,
    //            trans_tex_platform_info,
    //            Npulses * 4 * sizeof (float), Npulses * 4 * sizeof (float),
    //            1, cudaMemcpyHostToDevice));

    tex_platform_info.addressMode[0] = cudaAddressModeClamp;
    tex_platform_info.addressMode[1] = cudaAddressModeClamp;
    tex_platform_info.filterMode = cudaFilterModePoint;
    tex_platform_info.normalized = false; // access with normalized texture coordinates

    MY_CUDA_SAFE_CALL(cudaBindTextureToArray(tex_platform_info, array_tex_platform_info, float4desc));

    // Set up input projections texture
    float2* projections = format_complex_to_columns(phd, Nrangebins, Npulses);

    cudaChannelFormatDesc float2desc = cudaCreateChannelDesc<float2>();
    cudaArray* cu_proj;

    MY_CUDA_SAFE_CALL(cudaMallocArray(&cu_proj, &float2desc, Npulses, Nrangebins));
    MY_CUDA_SAFE_CALL(cudaMemcpyToArray(cu_proj, 0, 0, projections,
            Npulses * Nrangebins * 2 * sizeof (float), cudaMemcpyHostToDevice));
    //    MY_CUDA_SAFE_CALL(cudaMemcpy2DToArray(cu_proj, 0, 0,
    //            projections,
    //            Npulses * Nrangebins * 2 * sizeof (float), Npulses * Nrangebins * 2 * sizeof (float),
    //            1, cudaMemcpyHostToDevice));

    tex_projections.addressMode[0] = cudaAddressModeClamp;
    tex_projections.addressMode[1] = cudaAddressModeClamp;
    tex_projections.filterMode = cudaFilterModeLinear;
    tex_projections.normalized = false; // access with normalized texture coordinates

    MY_CUDA_SAFE_CALL(cudaBindTextureToArray(tex_projections, cu_proj, float2desc));

    // Set up and run the kernel
    dim3 dimBlock(blockwidth, blockheight, 1);
    dim3 dimGrid(Nx_pix / blockwidth, num_output_rows / blockheight);

    float delta_pixel_x = (right - left) / (Nx_pix - 1);
    float delta_pixel_y = (top - bottom) / (Ny_pix - 1);
    //float r_start_pre = (c__4_delta_freq * (float) total_proj_length / ((float) total_proj_length - 1.0f));

    float* device_start_frequencies;
    MY_CUDA_SAFE_CALL(cudaMalloc((void**) &device_start_frequencies, sizeof (float)*Npulses));
    MY_CUDA_SAFE_CALL(cudaMemcpy(device_start_frequencies, start_frequencies, sizeof (float)*Npulses, cudaMemcpyHostToDevice));


    clock_t c0, c1;
    c0 = clock();

    float * debug_1, * debug_2, *debug_3, *debug_4;

#if ZEROCOPY
    MY_CUDA_SAFE_CALL(cudaHostAlloc((void**) &out_image, num_out_bytes,
            cudaHostAllocMapped));

    float2 * device_pointer;
    MY_CUDA_SAFE_CALL(cudaHostGetDevicePointer((void **) &device_pointer,
            (void *) out_image, 0));

    backprojection_loop << <dimGrid, dimBlock>>>(device_pointer, Npulses, Ny_pix,
            delta_pixel_x, delta_pixel_y,
            total_proj_length, 0, start_output_index,
            c__4_delta_freq, device_start_frequencies, left, bottom, min_eff_idx, trans_tex_platform_info,
            debug_1, debug_2, debug_3, debug_4,
            0, 0);
#else

    MY_CUDA_SAFE_CALL(cudaMalloc((void**) &out_image, num_out_bytes));

    backprojection_loop << <dimGrid, dimBlock>>>(out_image, Npulses, Ny_pix,
            delta_pixel_x, delta_pixel_y,
            total_proj_length, 0, start_output_index,
            c__4_delta_freq, device_start_frequencies, left, bottom, min_eff_idx, device_tex_platform_info,
            debug_1, debug_2, debug_3, debug_4, 0, 0);
#endif



    cudaError_t this_error = cudaGetLastError();
    if (this_error != cudaSuccess) {
        printf("\nERROR: cudaGetLastError did NOT return success! DO NOT TRUST RESULTS!\n");
        printf("         '%s'\n", cudaGetErrorString(this_error));
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
        printf("\nERROR: threads did NOT synchronize! DO NOT TRUST RESULTS!\n\n");
    c1 = clock();
    printf("INFO: CUDA-mex kernel took %f s\n", (float) (c1 - c0) / CLOCKS_PER_SEC);

#if ZEROCOPY
    from_gpu_complex_to_bp_complex_split(out_image, host_output_image, num_output_rows * Ny_pix);
    MY_CUDA_SAFE_CALL(cudaFreeHost(out_image));
#else
    float2 * host_data = (float2 *) malloc(num_out_bytes);
    //double start_t = -ms_walltime();
    MY_CUDA_SAFE_CALL(cudaMemcpy(host_data, out_image, num_out_bytes, cudaMemcpyDeviceToHost));
    //printf("MEMCPY,%lf\n", (start_t + ms_walltime()));
    from_gpu_complex_to_bp_complex_split(host_data, host_output_image, num_output_rows
            * Ny_pix);
    free(host_data);
    cudaFree(out_image);
#endif
    cudaFree(device_start_frequencies);
    free(trans_tex_platform_info);
    free(projections);

    cudaFreeArray(array_tex_platform_info);
    cudaFree(device_tex_platform_info);
    cudaFreeArray(cu_proj);

    MY_CUDA_SAFE_CALL(cudaDeviceReset());

}

//__global__ void testing_platform_tex(float * x, float * y, float * z, float * w, float num)
//{
//    float4 foo = tex1D(tex_platform_info, num);
//    x[0] = foo.x;
//    y[0] = foo.y;
//    z[0] = foo.z;
//    w[0] = foo.w;
//}
//
//__global__ void testing_platform(float4 * plat, float * xx, float * yy, float * zz, float * ww, int num)
//{
//    float4 foo = plat[num];
//    xx[0] = foo.x;
//    yy[0] = foo.y;
//    zz[0] = foo.z;
//    ww[0] = foo.w;
//}
//
//__global__ void testing_proj_tex(float * re, float * im, float xx, float yy)
//{
//    float2 foo = tex2D(tex_projections, xx, yy); // x: proj num, y: rbin
//    re[0] = foo.x;
//    im[0] = foo.y;
//}
//
//__global__ void testing_r(float xpixel, float ypixel, float xa, float ya, float za, float * R) 
//{
//    (*R) = ( CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + 
//            __fmul_rn(za, za));
//}

