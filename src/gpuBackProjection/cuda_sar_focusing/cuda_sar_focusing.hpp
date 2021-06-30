/* 
 * File:   gpu_sar_focusing.hpp
 * Author: arwillis
 *
 * Created on June 20, 2021, 1:25 PM
 */

#ifndef GPU_SAR_FOCUSING_HPP
#define GPU_SAR_FOCUSING_HPP

#include <memory>
#include <unordered_map>
#include <iostream>
#include <numeric>

// CUDA includes
#include <helper_cuda.h>

#include "cuda_BackProjectionKernels.cuh"

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

class CUDADevice_ArrayMappedMemory {
public:
    typedef std::shared_ptr<CUDADevice_ArrayMappedMemory> Ptr;
    void* host_mem;
    void* dev_mem;
    size_t size;

    CUDADevice_ArrayMappedMemory() : dev_mem(nullptr), size(0), host_mem(nullptr) {
    }

    CUDADevice_ArrayMappedMemory(void* _host_mem, size_t _size) : dev_mem(nullptr), size(_size), host_mem(_host_mem) {
    }

    virtual ~CUDADevice_ArrayMappedMemory() {
    }

    CUDADevice_ArrayMappedMemory::Ptr create() {
        return std::make_shared<CUDADevice_ArrayMappedMemory>();
    }

    bool isEmpty() {
        return (dev_mem == nullptr && host_mem == nullptr) || size == 0;
    }

    template<typename __Tp> __Tp* getDeviceMemPointer() {
        return (__Tp*) dev_mem;
    }

    template<typename __Tp> __Tp* getHostMemPointer() {
        return (__Tp*) dev_mem;
    }

    friend std::ostream& operator<<(std::ostream& output, const CUDADevice_ArrayMappedMemory& gpu_mm);

};

inline std::ostream& operator<<(std::ostream& output, const CUDADevice_ArrayMappedMemory& gpu_mem) {
    output << "{ size=" << gpu_mem.size << " bytes hostptr=" << gpu_mem.host_mem << " devptr=" << gpu_mem.dev_mem << " }";
    return output;
}

class GPUMemoryManager {
public:
    typedef std::shared_ptr<GPUMemoryManager> Ptr;
    typedef std::unordered_map<std::string, CUDADevice_ArrayMappedMemory>::iterator mmap_iterator;

    int deviceId;
    int blockwidth, blockheight;
    cudaDeviceProp props;

    std::unordered_map<std::string, CUDADevice_ArrayMappedMemory> gpu_mmap;

    int createOnDevice(std::string name, size_t size_in_bytes) {
        if (gpu_mmap.find(name) != gpu_mmap.end()) {
            std::cout << "createOnDevice::Failure to export data with the name \"" + name + "\". The name already exists in the GPU memory map." << std::endl;
            return EXIT_FAILURE;
        }
        CUDADevice_ArrayMappedMemory gpu_mem(nullptr, size_in_bytes);
        MY_CUDA_SAFE_CALL(cudaMalloc(&gpu_mem.dev_mem, size_in_bytes));
        gpu_mmap[name] = gpu_mem;
        return EXIT_SUCCESS;
    }

    int copyToDevice(std::string name, void* src_data, size_t size_in_bytes) {
        if (gpu_mmap.find(name) == gpu_mmap.end()) {
            if (createOnDevice(name, size_in_bytes) == EXIT_FAILURE) {
                return EXIT_FAILURE;
            }
        }
        std::unordered_map<std::string, CUDADevice_ArrayMappedMemory>::iterator it = gpu_mmap.find(name);
        CUDADevice_ArrayMappedMemory& gpu_mem = it->second;
        if (!src_data) {
            std::cout << "copyToDevice::Failure to copy data from host source address; it is a nullptr." << std::endl;
            return EXIT_FAILURE;
        }
        gpu_mem.host_mem = src_data;
        MY_CUDA_SAFE_CALL(cudaMemcpy(gpu_mem.dev_mem, src_data, size_in_bytes, cudaMemcpyHostToDevice));
        return EXIT_SUCCESS;
    }

    template<typename __Tp> __Tp* getHostMemPointer(std::string name) {
        mmap_iterator it = gpu_mmap.find(name);
        if (it == gpu_mmap.end()) {
            std::cout << "getHostMemPointer::Failure to find GPU entry with the name \"" + name + "\". It does not exist in the GPU memory map." << std::endl;
        }
        return (__Tp*) it->second.host_mem;
    }

    template<typename __Tp> __Tp* getDeviceMemPointer(std::string name) {
        mmap_iterator it = gpu_mmap.find(name);
        if (it == gpu_mmap.end()) {
            std::cout << "getDeviceMemPointer::Failure to find GPU entry with the name \"" + name + "\". It does not exist in the GPU memory map." << std::endl;
        }
        return (__Tp*) it->second.dev_mem;
    }

    int copyFromDevice(std::string name, void* dst_data, size_t size_in_bytes) {
        mmap_iterator it = gpu_mmap.find(name);
        if (it == gpu_mmap.end()) {
            std::cout << "copyFromDevice::Failure to find GPU entry with the name \"" + name + "\". It does not exist in the GPU memory map." << std::endl;
            return EXIT_FAILURE;
        }
        CUDADevice_ArrayMappedMemory& gpu_mem = it->second;
        if (!dst_data) {
            std::cout << "copyFromDevice::Failure to copy. The host destination address is a nullptr." << std::endl;
            return EXIT_FAILURE;
        }
        if (!gpu_mem.dev_mem) {
            std::cout << "copyFromDevice::Failure to copy. The device source address is a nullptr." << std::endl;
            return EXIT_FAILURE;
        }
        if (size_in_bytes > gpu_mem.size) {
            std::cout << "copyFromDevice::Failure to copy. The destination address memory size is larger than the device source size." << std::endl;
            return EXIT_FAILURE;
        } else if (size_in_bytes < gpu_mem.size) {
            std::cout << "copyFromDevice::Warning. The destination address memory size is smaller than the device source size." << std::endl;
        }
        MY_CUDA_SAFE_CALL(cudaMemcpy(dst_data, gpu_mem.dev_mem, size_in_bytes, cudaMemcpyDeviceToHost));
        return EXIT_SUCCESS;
    }

    int freeGPUMemory(std::string name) {
        mmap_iterator it = gpu_mmap.find(name);
        if (it == gpu_mmap.end()) {
            std::cout << "freeGPUMemory::Failure to free. GPU memory with the name \"" + name + "\" does not exist in the GPU memory map." << std::endl;
            return EXIT_FAILURE;
        }
        CUDADevice_ArrayMappedMemory& gpu_mem = it->second;
        if (!gpu_mem.dev_mem) {
            std::cout << "freeGPUMemory::Failure to free. GPU memory with the name \"" + name + "\" has device memory address = nullptr." << std::endl;
        }
        cudaFree(gpu_mem.dev_mem);
        gpu_mem.dev_mem = nullptr;
        if (gpu_mem.isEmpty()) {
            gpu_mmap.erase(name);
        }
        return EXIT_SUCCESS;
    }

    int freeHostMemory(std::string name) {
        mmap_iterator it = gpu_mmap.find(name);
        if (it == gpu_mmap.end()) {
            std::cout << "freeHostMemory::Failure to free. Host memory with the name \"" + name + "\" does not exist in the GPU memory map." << std::endl;
            return EXIT_FAILURE;
        }
        CUDADevice_ArrayMappedMemory& gpu_mem = it->second;
        if (!gpu_mem.host_mem) {
            std::cout << "freeHostMemory::Failure to free. Host memory with the name \"" + name + "\" has host memory address = nullptr." << std::endl;
        }
        free(gpu_mem.host_mem);
        gpu_mem.host_mem = nullptr;
        if (gpu_mem.isEmpty()) {
            gpu_mmap.erase(name);
        }
        return EXIT_SUCCESS;
    }

    int freeAllResources(std::string name) {
        if (freeGPUMemory(name) != EXIT_SUCCESS ||
                freeHostMemory(name) != EXIT_SUCCESS) {
            std::cout << "freeAllResources::Failed to unallocated all resources." << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    int freeAllResources() {
        int result = EXIT_SUCCESS;
        for (const auto it : gpu_mmap) {
            std::cout << "Freeing GPU-Host memory for array " << it.first << " : " << it.second.size << " bytes freed." << std::endl;
            if (freeAllResources(it.first) == EXIT_FAILURE)
                result = EXIT_FAILURE;
        }
        return result;
    }

    GPUMemoryManager() : deviceId(-1) {
    }

    virtual ~GPUMemoryManager() {
    }

    GPUMemoryManager::Ptr create() {
        return std::make_shared<GPUMemoryManager>();
    }

    friend std::ostream& operator<<(std::ostream& output, const GPUMemoryManager& gpu_mm);

};

inline std::ostream& operator<<(std::ostream& output, const GPUMemoryManager& gpu_mm) {
    std::cout << "GPU-Host memory map: " << std::endl;
    for (const auto it : gpu_mm.gpu_mmap) {
        output << "\tmap[\"" << it.first << "\"] = " << it.second << std::endl;
    }
    return output;
}

// future contents of gpu_sar_focusing_fft.hpp
#include <cufft.h>
//#include <cufftw.h>

// On the cufftComplex data type:
// from: https://stackoverflow.com/questions/13535182/copying-data-to-cufftcomplex-data-struct
//
// From nVidia : " cufftComplex is a single‐precision, floating‐point complex 
// data type that consists of interleaved real and imaginary components."
// The storage layout of complex data types in CUDA is compatible with the 
// layout defined for complex types in Fortran and C++, i.e. as a structure 
// with the real part followed by imaginary part.
//

void cufft_engine(cufftComplex *dev_signal, int N_fft, int N_batch, int DIR) {
    // CUFFT plan
    cufftHandle plan;
    if (cufftPlan1d(&plan, N_fft, CUFFT_C2C, N_batch) != CUFFT_SUCCESS) {
        std::cout << "CUFFT error: Plan creation failed" << std::endl;
        return;
    }
    //int istride = 1, ostride = 1;
    //int o
    //cufftPlanMany(&plan, 1, N_fft, NULL, istride, idist, 
    //    NULL, ostride, odist, CUFFT_C2C, N_batch);
    // Transform signal and kernel
    printf("Transforming signal cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *) dev_signal, (cufftComplex *) dev_signal, DIR);
}

void cufft(cufftComplex *dev_signal, int N_fft, int N_batch) {
    cufft_engine(dev_signal, N_fft, N_batch, CUFFT_FORWARD);
}

void cuifft(cufftComplex *dev_signal, int N_fft, int N_batch) {
    cufft_engine(dev_signal, N_fft, N_batch, CUFFT_INVERSE);
}

// functions that format data for export to GPU (perhaps not required)

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

// GPU initialization

int initialize_GPUMATLAB(int& deviceId) {
    // This will pick the best possible CUDA capable device
    //    devID = findCudaDevice(argc, (const char **)argv); 
    if (deviceId == -1) {
        // Otherwise pick the device with highest Gflops/s
        deviceId = gpuGetMaxGflopsDeviceId();
        MY_CUDA_SAFE_CALL(cudaSetDevice(deviceId));
    }
    return EXIT_SUCCESS;
}

template<typename __nTp1, typename __nTp2>
int initialize_CUDAResources(const SAR_Aperture<__nTp1>& sar_data,
        const SAR_ImageFormationParameters<__nTp2>& sar_img_params,
        GPUMemoryManager& cuda_res) {
    initialize_GPUMATLAB(cuda_res.deviceId);
    cuda_res.blockwidth = BLOCKWIDTH;
    cuda_res.blockheight = BLOCKHEIGHT;

    //Get GPU information
    //    MY_CUDA_SAFE_CALL(cudaGetDevice(&deviceId));
    MY_CUDA_SAFE_CALL(cudaSetDevice(cuda_res.deviceId));
    MY_CUDA_SAFE_CALL(cudaGetDeviceProperties(&cuda_res.props, cuda_res.deviceId));
    //MY_CUDA_SAFE_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId));
    //MY_CUDA_SAFE_CALL(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceId));
    std::cout << "Device " << cuda_res.deviceId << ": \"" << cuda_res.props.name
            << " has NVIDIA \"" << _ConvertSMVer2ArchName(cuda_res.props.major, cuda_res.props.minor) << "architecture"
            << "\" with compute capability " << cuda_res.props.major << "." << cuda_res.props.minor << "." << std::endl;
    std::cout << "CUDA parameters not provided. Auto-selecting:" << std::endl
            << "\tdevice                " << cuda_res.deviceId << std::endl
            << "\tblockwidth            " << cuda_res.blockwidth << std::endl
            << "\tblockheight           " << cuda_res.blockheight << std::endl;
    //<< "\ttexturePitchAlignment " << cuda_res.props.texturePitchAlignment << std::endl;

#if ZEROCOPY
    // We will want ZEROCOPY code for Xavier and newer architecture platforms
    // https://developer.ridgerun.com/wiki/index.php?title=NVIDIA_CUDA_Memory_Management
    MY_CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif
    // We will want UNIFIED MEMORY code for Maxwell architecture platforms
    cuda_res.copyToDevice("Ant_x", (void *) &sar_data.Ant_x.data[0],
            sar_data.Ant_x.data.size() * sizeof (sar_data.Ant_x.data[0]));
    cuda_res.copyToDevice("Ant_y", (void *) &sar_data.Ant_y.data[0],
            sar_data.Ant_y.data.size() * sizeof (sar_data.Ant_y.data[0]));
    cuda_res.copyToDevice("Ant_z", (void *) &sar_data.Ant_z.data[0],
            sar_data.Ant_z.data.size() * sizeof (sar_data.Ant_z.data[0]));
    cuda_res.copyToDevice("slant_range", (void *) &sar_data.slant_range.data[0],
            sar_data.slant_range.data.size() * sizeof (sar_data.slant_range.data[0]));
    cuda_res.copyToDevice("sampleData", (void *) &sar_data.sampleData.data[0],
            sar_data.sampleData.data.size() * sizeof (sar_data.sampleData.data[0]));
    cuda_res.copyToDevice("startF", (void *) &sar_data.startF.data[0],
            sar_data.startF.data.size() * sizeof (sar_data.startF.data[0]));
    cuda_res.copyToDevice("sar_image_params", (void *) &sar_img_params,
            sizeof (sar_img_params));

    int num_img_bytes = 2 * sizeof (float) * sar_img_params.N_x_pix * sar_img_params.N_y_pix;
#if ZEROCOPY
    MY_CUDA_SAFE_CALL(cudaHostAlloc((void**) &cuda_res.out_image, num_img_bytes,
            cudaHostAllocMapped));

    float2 * device_pointer;
    MY_CUDA_SAFE_CALL(cudaHostGetDevicePointer((void **) &device_pointer,
            (void *) cuda_res.out_image, 0));
#else
    cuda_res.createOnDevice("output_image", num_img_bytes);
#endif
    return EXIT_SUCCESS;
}

template<typename __nTp1, typename __nTp2>
int finalize_CUDAResources(const SAR_Aperture<__nTp1>& sar_data,
        const SAR_ImageFormationParameters<__nTp2>& sar_img_params,
        GPUMemoryManager& cuda_res) {
    cudaError_t this_error = cudaGetLastError();
    if (this_error != cudaSuccess) {
        printf("\nERROR: cudaGetLastError did NOT return success! DO NOT TRUST RESULTS!\n");
        printf("         '%s'\n", cudaGetErrorString(this_error));
    }

#if ZEROCOPY
    MY_CUDA_SAFE_CALL(cudaFreeHost(cuda_res.out_image));
#else
    cuda_res.freeGPUMemory("output_image");
#endif
    cuda_res.freeGPUMemory("sar_image_params");
    cuda_res.freeGPUMemory("Ant_x");
    cuda_res.freeGPUMemory("Ant_y");
    cuda_res.freeGPUMemory("Ant_z");
    cuda_res.freeGPUMemory("slant_range");
    cuda_res.freeGPUMemory("startF");
    cuda_res.freeGPUMemory("sampleData");

    MY_CUDA_SAFE_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

//#include "cufftShift_1D_IP.cuh"

template <typename __nTp, typename __nTpParams>
void cuda_focus_SAR_image(const SAR_Aperture<__nTp>& sar_data,
        const SAR_ImageFormationParameters<__nTpParams>& sar_image_params,
        CArray<__nTp>& output_image) {

    // Display maximum scene size and resolution
    std::cout << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << sar_image_params.max_Wy_m << " m range, "
            << sar_image_params.max_Wx_m << " m cross-range" << std::endl;
    std::cout << "Resolution:  " << std::fixed << std::setprecision(2) << sar_image_params.slant_rangeResolution << "m range, "
            << sar_image_params.azimuthResolution << " m cross-range" << std::endl;
    GPUMemoryManager cuda_res;

    if (initialize_GPUMATLAB(cuda_res.deviceId) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Could not initialize the GPU. Exiting..." << std::endl;
        return;
    }
    if (initialize_CUDAResources(sar_data, sar_image_params, cuda_res) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Problem found initializing resources on the GPU. Exiting..." << std::endl;
        return;
    }
    std::cout << cuda_res << std::endl;
    int numSamples = sar_data.sampleData.data.size();
    //    Complex<__nTp> devResult1[numSamples];
    //    cuda_res.copyFromDevice("sampleData", &devResult1[0], numSamples * sizeof (Complex<__nTp>));
    //    for (int i = 0; i < 10; i++) {
    //        std::cout << "sampleData[" << i << "]=" << std::setprecision(7) << devResult1[i] << std::endl;
    //    }

    cuifft(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"), sar_image_params.N_fft, sar_data.numAzimuthSamples);

    //    Complex<__nTp> devResult[numSamples];
    //    cuda_res.copyFromDevice("sampleData", &devResult[0], numSamples * sizeof (Complex<__nTp>));
    //    for (int i = 0; i < 10; i++) {
    //        std::cout << "ifft(sampleData)[" << i << "]=" << std::setprecision(7) << devResult[i]/sar_image_params.N_fft << std::endl;
    //    }
    cufftShift_1D<cufftComplex>(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"), sar_image_params.N_fft);
    //    Complex<__nTp> devResult1[numSamples];
    //    cuda_res.copyFromDevice("sampleData", &devResult1[0], numSamples * sizeof (Complex<__nTp>));
    //    for (int i = 0; i < 10; i++) {
    //        std::cout << "fftshift(ifft(sampleData))[" << i << "]=" << std::setprecision(7) << devResult1[i]/sar_image_params.N_fft << std::endl;
    //    }
    //__nTp c__4_delta_freq = CLIGHT / (4.0 * sar_data.deltaF.data[0]);
    //__nTp pi_4_f0__clight = (PI * 4.0 * sar_data.startF.data[0]) / CLIGHT;
    //__nTp r_start_pre = (c__4_delta_freq * (float) sar_data.numRangeSamples / ((float) sar_data.numRangeSamples - 1.0f));
    //convert_f0(start_frequencies, Npulses);

    __nTp delta_x_m_per_pix = sar_image_params.Wx_m / (sar_image_params.N_x_pix - 1);
    __nTp delta_y_m_per_pix = sar_image_params.Wy_m / (sar_image_params.N_y_pix - 1);
    __nTp left_m = sar_image_params.x0_m - sar_image_params.Wx_m / 2;
    __nTp bottom_m = sar_image_params.y0_m - sar_image_params.Wy_m / 2;

    // Set up and run the kernel
    dim3 dimBlock(cuda_res.blockwidth, cuda_res.blockheight, 1);
    dim3 dimGrid(sar_image_params.N_x_pix / cuda_res.blockwidth,
            sar_image_params.N_y_pix / cuda_res.blockheight);

    clock_t c0, c1;
    c0 = clock();

#if ZEROCOPY
    backprojection_loop << <dimGrid, dimBlock>>>(cuda_res.getDeviceMemPointer<float2>("sampleData"),
            sar_data.numAzimuthSamples, sar_image_params.N_y_pix,
            delta_x, delta_y, sar_data.numRangeSamples, 0, 0,
            c__4_delta_freq, cuda_res.getDeviceMemPointer<float>("startF"),
            left, bottom, cuda_res.getDeviceMemPointer<float4>("platform_positions"), 0, 0,
            cuda_res.getDeviceMemPointer<float2>("output_image"));
#else
    backprojection_loop<__nTp> << <dimGrid, dimBlock>>>(cuda_res.getDeviceMemPointer<float2>("sampleData"),
            sar_data.numRangeSamples, sar_data.numAzimuthSamples,
            delta_x_m_per_pix, delta_y_m_per_pix,
            left_m, bottom_m, 0, 0,
            cuda_res.getDeviceMemPointer<__nTp>("Ant_x"),
            cuda_res.getDeviceMemPointer<__nTp>("Ant_y"),
            cuda_res.getDeviceMemPointer<__nTp>("Ant_z"),
            cuda_res.getDeviceMemPointer<__nTp>("slant_range"),
            cuda_res.getDeviceMemPointer<__nTp>("startF"),
            cuda_res.getDeviceMemPointer<SAR_ImageFormationParameters < __nTpParams >> ("sar_image_params"),
            cuda_res.getDeviceMemPointer<float2>("output_image"));
#endif
    c1 = clock();
    printf("INFO: CUDA-mex kernel took %f s\n", (float) (c1 - c0) / CLOCKS_PER_SEC);
    if (cudaDeviceSynchronize() != cudaSuccess)
        printf("\nERROR: threads did NOT synchronize! DO NOT TRUST RESULTS!\n\n");

#if ZEROCOPY
    from_gpu_complex_to_bp_complex_split(cuda_res.out_image, output_image, sar_image_params.N_x_pix * sar_image_params.N_y_pix);
#else
    int num_img_bytes = 2 * sizeof (float) * sar_image_params.N_x_pix * sar_image_params.N_y_pix;
    float2* host_data = (float2*) malloc(num_img_bytes);
    //double start_t = -ms_walltime();
    //MY_CUDA_SAFE_CALL(cudaMemcpy(host_data, cuda_res.out_image, cuda_res.num_out_bytes, cudaMemcpyDeviceToHost));
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
    std::cout << cuda_res << std::endl;
}

#endif /* GPU_SAR_FOCUSING_HPP */

