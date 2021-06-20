/* 
 * File:   gpu_sar_focusing.hpp
 * Author: arwillis
 *
 * Created on June 20, 2021, 1:25 PM
 */

#ifndef GPU_SAR_FOCUSING_HPP
#define GPU_SAR_FOCUSING_HPP

// CUDA includes
#include <helper_cuda.h>

#define MY_CUDA_SAFE_CALL_NO_SYNC( call) {                                   \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        printf( "Cuda error in file '%s' in line %i : %s.\n",                \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    }}
#define MY_CUDA_SAFE_CALL(call) MY_CUDA_SAFE_CALL_NO_SYNC(call);
#define MEXDEBUG      1
#define PI2 6.2831853071800f   /* 2*pi */
#define PI_4__CLIGHT (4.0f * PI / CLIGHT)

#define ASSUME_Z_0    1     /* Ignore consult_DEM() and assume height = 0. */
#define USE_FAST_MATH 0     /* Use __math() functions? */
#define USE_RSQRT     0

#define ZEROCOPY      0

class SAR_GPU_Resources {
public:
    int deviceId;
    int blockwidth, blockheight;
    cudaDeviceProp props;

    float4* trans_tex_platform_info;
    cudaArray* array_tex_platform_info;
    float4* device_tex_platform_info;

    float2* projections;
    cudaArray* cu_proj;

    float* device_start_frequencies;

    float2* out_image;
    int num_out_bytes;

    SAR_GPU_Resources() : deviceId(-1) {
    }

    virtual ~SAR_GPU_Resources() {
    }
};

/* Globals and externs */

/* Complex textures containing range profiles */
//texture<float2, 2, cudaReadModeElementType> tex_projections;

/* 4-elem. textures for x, y, z, r0 */
//texture<float4, 1, cudaReadModeElementType> tex_platform_info;

template<typename __nTp>
void from_gpu_complex_to_bp_complex_split(float2* data, CArray<__nTp>& out, int size) {
    int i;
    for (i = 0; i < size; i++) {
        out[i]._M_real = data[i].x;
        out[i]._M_imag = data[i].y;
    }
}

template<typename __nTp>
float2* format_complex_to_columns(const Complex<__nTp>* a, int width_orig, int height_orig) {
    float2* out = (float2*) malloc(width_orig * height_orig * sizeof (float2));
    int i, j;
    for (i = 0; i < height_orig; i++) {
        int origOffset = i * width_orig;
        for (j = 0; j < width_orig; j++) {
            int newOffset = j * height_orig;
            out[newOffset + i].x = a[origOffset + j].real();
            out[newOffset + i].y = a[origOffset + j].imag();
        }
    }
    return out;
}

float4* format_x_y_z_r(const float* x, const float* y, const float* z, const float* r, const int size) {
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

int initialize_GPUMATLAB(int& deviceId) {

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // This will pick the best possible CUDA capable device
    //    devID = findCudaDevice(argc, (const char **)argv); 
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
    return EXIT_SUCCESS;
}

template<typename __nTp1, typename __nTp2>
int initialize_CUDAResources(const SAR_Aperture<__nTp1>& sar_data,
        const SAR_ImageFormationParameters<__nTp2>& sar_img_params,
        SAR_GPU_Resources& cuda_res) {

    initialize_GPUMATLAB(cuda_res.deviceId);
    cuda_res.blockwidth = BLOCKWIDTH;
    cuda_res.blockheight = BLOCKHEIGHT;

    std::cout << "CUDA parameters not provided. Auto-selecting:" << std::endl
            << "\tdevice      " << cuda_res.deviceId << std::endl
            << "\tblockwidth  " << cuda_res.blockwidth << std::endl
            << "\tblockheight " << cuda_res.blockheight << std::endl;

    //Get GPU information
    //    MY_CUDA_SAFE_CALL(cudaGetDevice(&deviceId));
    MY_CUDA_SAFE_CALL(cudaSetDevice(cuda_res.deviceId));
    MY_CUDA_SAFE_CALL(cudaGetDeviceProperties(&cuda_res.props, cuda_res.deviceId));
    std::cout << "Device " << cuda_res.deviceId << ": \"" << cuda_res.props.name
            << "\" with compute " << cuda_res.props.major << "." << cuda_res.props.minor << " capability" << std::endl;

#if ZEROCOPY
    // We will want ZEROCOPY code for Xavier and newer architecture platforms
    // https://developer.ridgerun.com/wiki/index.php?title=NVIDIA_CUDA_Memory_Management
    MY_CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif
    // We will want UNIFIED MEMORY code for Maxwell architecture platforms

    // Set up platform data texture
    cuda_res.trans_tex_platform_info = format_x_y_z_r(&sar_data.Ant_x.data[0],
            &sar_data.Ant_y.data[0],
            &sar_data.Ant_z.data[0],
            &sar_data.slant_range.data[0],
            sar_data.numAzimuthSamples);
    cudaChannelFormatDesc float4desc = cudaCreateChannelDesc<float4>();
    //float4* device_tex_platform_info;
    //cudaArray* gpuInfo.array_tex_platform_info;

    MY_CUDA_SAFE_CALL(cudaMallocArray(&cuda_res.array_tex_platform_info,
            &float4desc, sar_data.numAzimuthSamples, 1));
    MY_CUDA_SAFE_CALL(cudaMemcpyToArray(cuda_res.array_tex_platform_info, 0, 0,
            cuda_res.trans_tex_platform_info, sar_data.numAzimuthSamples * 4 * sizeof (float),
            cudaMemcpyHostToDevice));

    MY_CUDA_SAFE_CALL(cudaMalloc(&cuda_res.device_tex_platform_info,
            sizeof (float4) * sar_data.numAzimuthSamples));
    MY_CUDA_SAFE_CALL(cudaMemcpy(cuda_res.device_tex_platform_info,
            cuda_res.trans_tex_platform_info, sar_data.numAzimuthSamples * sizeof (float4),
            cudaMemcpyHostToDevice));
    //    MY_CUDA_SAFE_CALL(cudaMemcpy2DToArray(array_tex_platform_info, 0, 0,
    //            trans_tex_platform_info,
    //            Npulses * 4 * sizeof (float), Npulses * 4 * sizeof (float),
    //            1, cudaMemcpyHostToDevice));

    tex_platform_info.addressMode[0] = cudaAddressModeClamp;
    tex_platform_info.addressMode[1] = cudaAddressModeClamp;
    tex_platform_info.filterMode = cudaFilterModePoint;
    tex_platform_info.normalized = false; // access with normalized texture coordinates

    MY_CUDA_SAFE_CALL(cudaBindTextureToArray(tex_platform_info, cuda_res.array_tex_platform_info, float4desc));

    // Set up input projections texture
    cuda_res.projections = format_complex_to_columns(&sar_data.sampleData.data[0],
            sar_data.numRangeSamples, sar_data.numAzimuthSamples);

    cudaChannelFormatDesc float2desc = cudaCreateChannelDesc<float2>();

    MY_CUDA_SAFE_CALL(cudaMallocArray(&cuda_res.cu_proj, &float2desc, sar_data.numAzimuthSamples, sar_data.numRangeSamples));
    MY_CUDA_SAFE_CALL(cudaMemcpyToArray(cuda_res.cu_proj, 0, 0, cuda_res.projections,
            sar_data.numAzimuthSamples * sar_data.numRangeSamples * 2 * sizeof (float), cudaMemcpyHostToDevice));
    //    MY_CUDA_SAFE_CALL(cudaMemcpy2DToArray(cu_proj, 0, 0,
    //            projections,
    //            Npulses * Nrangebins * 2 * sizeof (float), Npulses * Nrangebins * 2 * sizeof (float),
    //            1, cudaMemcpyHostToDevice));

    tex_projections.addressMode[0] = cudaAddressModeClamp;
    tex_projections.addressMode[1] = cudaAddressModeClamp;
    tex_projections.filterMode = cudaFilterModeLinear;
    tex_projections.normalized = false; // access with normalized texture coordinates

    MY_CUDA_SAFE_CALL(cudaBindTextureToArray(tex_projections, cuda_res.cu_proj, float2desc));

    MY_CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_res.device_start_frequencies,
            sizeof (float) * sar_data.numAzimuthSamples));
    MY_CUDA_SAFE_CALL(cudaMemcpy(cuda_res.device_start_frequencies, &sar_data.startF.data[0],
            sizeof (float) * sar_data.numAzimuthSamples, cudaMemcpyHostToDevice));

    cuda_res.num_out_bytes = 2 * sizeof (float) * sar_img_params.N_x_pix * sar_img_params.N_y_pix;
#if ZEROCOPY
    MY_CUDA_SAFE_CALL(cudaHostAlloc((void**) &cuda_res.out_image, cuda_res.num_out_bytes,
            cudaHostAllocMapped));

    float2 * device_pointer;
    MY_CUDA_SAFE_CALL(cudaHostGetDevicePointer((void **) &device_pointer,
            (void *) cuda_res.out_image, 0));
#else
    MY_CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_res.out_image, cuda_res.num_out_bytes));
#endif
    return EXIT_SUCCESS;
}

template<typename __nTp1, typename __nTp2>
int finalize_CUDAResources(const SAR_Aperture<__nTp1>& sar_data,
        const SAR_ImageFormationParameters<__nTp2>& sar_img_params,
        SAR_GPU_Resources& cuda_res) {
    cudaError_t this_error = cudaGetLastError();
    if (this_error != cudaSuccess) {
        printf("\nERROR: cudaGetLastError did NOT return success! DO NOT TRUST RESULTS!\n");
        printf("         '%s'\n", cudaGetErrorString(this_error));
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
        printf("\nERROR: threads did NOT synchronize! DO NOT TRUST RESULTS!\n\n");

#if ZEROCOPY
    MY_CUDA_SAFE_CALL(cudaFreeHost(cuda_res.out_image));
#else
    cudaFree(cuda_res.out_image);
#endif

    cudaFree(cuda_res.device_start_frequencies);
    free(cuda_res.trans_tex_platform_info);
    free(cuda_res.projections);

    cudaFreeArray(cuda_res.array_tex_platform_info);
    cudaFree(cuda_res.device_tex_platform_info);
    cudaFreeArray(cuda_res.cu_proj);

    MY_CUDA_SAFE_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

template <typename __nTp, typename __nTpParams>
void cuda_focus_SAR_image(const SAR_Aperture<__nTp>& sar_data,
        const SAR_ImageFormationParameters<__nTpParams>& sar_image_params,
        CArray<__nTp>& output_image) {

    // Display maximum scene size and resolution
    std::cout << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << sar_image_params.max_Wy_m << " m range, "
            << sar_image_params.max_Wx_m << " m cross-range" << std::endl;
    std::cout << "Resolution:  " << std::fixed << std::setprecision(2) << sar_image_params.slant_rangeResolution << "m range, "
            << sar_image_params.azimuthResolution << " m cross-range" << std::endl;
    SAR_GPU_Resources cuda_res;
    if (initialize_GPUMATLAB(cuda_res.deviceId) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Could not initialize the GPU. Exiting..." << std::endl;
        return;
    }
    if (initialize_CUDAResources(sar_data, sar_image_params, cuda_res) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Problem found initializing resources on the GPU. Exiting..." << std::endl;
        return;
    }

    __nTp c__4_delta_freq = CLIGHT / (4.0 * sar_data.deltaF.data[0]);
    __nTp pi_4_f0__clight = (PI * 4.0 * sar_data.startF.data[0]) / CLIGHT;
    //convert_f0(start_frequencies, Npulses);
    __nTp delta_x = sar_image_params.Wx_m / (sar_image_params.N_x_pix - 1);
    __nTp delta_y = sar_image_params.Wy_m / (sar_image_params.N_y_pix - 1);
    __nTp left = sar_image_params.x0_m - sar_image_params.Wx_m / 2;
    __nTp bottom = sar_image_params.y0_m - sar_image_params.Wy_m / 2;
    // Set up and run the kernel
    dim3 dimBlock(cuda_res.blockwidth, cuda_res.blockheight, 1);
    dim3 dimGrid(sar_image_params.N_x_pix / cuda_res.blockwidth,
            sar_image_params.N_y_pix / cuda_res.blockheight);
    //float r_start_pre = (c__4_delta_freq * (float) total_proj_length / ((float) total_proj_length - 1.0f));

    clock_t c0, c1;
    c0 = clock();
    float * debug_1, * debug_2, *debug_3, *debug_4;

#if ZEROCOPY
    backprojection_loop << <dimGrid, dimBlock>>>(device_pointer, Npulses, Ny_pix,
            delta_x, delta_y, sar_data.numRangeSamples, 0, 0,
            c__4_delta_freq, cuda_resources.device_start_frequencies,
            left, bottom, cuda_res.trans_tex_platform_info,
            debug_1, debug_2, debug_3, debug_4, 0, 0);
#else
    backprojection_loop << <dimGrid, dimBlock>>>(cuda_res.out_image,
            sar_data.numAzimuthSamples, sar_image_params.N_y_pix,
            delta_x, delta_y, sar_data.numRangeSamples, 0, 0,
            c__4_delta_freq, cuda_res.device_start_frequencies,
            left, bottom, cuda_res.device_tex_platform_info,
            debug_1, debug_2, debug_3, debug_4, 0, 0);
#endif
    c1 = clock();
    printf("INFO: CUDA-mex kernel took %f s\n", (float) (c1 - c0) / CLOCKS_PER_SEC);

#if ZEROCOPY
    from_gpu_complex_to_bp_complex_split(cuda_res.out_image, output_image, sar_image_params.N_x_pix * sar_image_params.N_y_pix);
#else
    float2* host_data = (float2 *) malloc(cuda_res.num_out_bytes);
    //double start_t = -ms_walltime();
    MY_CUDA_SAFE_CALL(cudaMemcpy(host_data, cuda_res.out_image, cuda_res.num_out_bytes, cudaMemcpyDeviceToHost));
    //printf("MEMCPY,%lf\n", (start_t + ms_walltime()));
    from_gpu_complex_to_bp_complex_split(host_data,
            output_image,
            sar_image_params.N_x_pix * sar_image_params.N_y_pix);
    free(host_data);
#endif

    if (finalize_CUDAResources(sar_data, sar_image_params, cuda_res) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Problem found de-allocating and free resources on the GPU. Exiting..." << std::endl;
        return;
    }
}

#endif /* GPU_SAR_FOCUSING_HPP */

