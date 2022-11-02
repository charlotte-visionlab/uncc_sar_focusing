/*
 * Copyright (C) 2022 Andrew R. Willis
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

// Standard Library includes
#include <iomanip>
#include <sstream>
#include <fstream>

#include <third_party/log.h>
#include <third_party/cxxopts.hpp>

#include <cuGridSearch.cuh>

// this declaration needs to be in any C++ compiled target for CPU
//#define CUDAFUNCTION

#include <charlotte_sar_api.hpp>
#include <uncc_sar_globals.hpp>

#include <uncc_sar_focusing.hpp>
#include <uncc_sar_matio.hpp>

#include "../gpuBackProjection/cuda_sar_focusing/cuda_sar_focusing.hpp"

#include "gridSearchErrorFunctions.cuh"

typedef float NumericType;

#define grid_dimension 6        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
typedef float grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef float pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

// // TODO: THIS WILL NEED TO BE CHANGED TO FIT THE ERROR FUNCTION (Look at changes to error function)
typedef func_byvalue_t<func_precision, grid_precision, grid_dimension, 
        cufftComplex*,
        int, int,
        NumericType, NumericType,
        NumericType, NumericType,
        NumericType, NumericType,
        NumericType*,
        NumericType*,
        NumericType*,
        NumericType*,
        NumericType*,
        SAR_ImageFormationParameters<NumericType>*,
        NumericType* > image_err_func_byvalue;

// // TODO: THIS WILL ALSO NEED TO BE CHANGED TO FIT THE ERROR FUNCTION
__device__ image_err_func_byvalue dev_func_byvalue_ptr = kernelWrapper<func_precision, grid_precision, grid_dimension, NumericType>;

pixel_precision imageA_data[6 * 6] = {0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 1, 1, 0, 0,
                                      0, 0, 1, 1, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0};

pixel_precision imageB_data[6 * 6] = {0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 1, 1,
                                      0, 0, 0, 0, 1, 1};

template<typename __nTp>
std::vector<__nTp> vectorDiff(std::vector<__nTp> values) {
    std::vector<__nTp> temp;
    for (int i = 0; i < values.size()-1; i++)
        temp.push_back(values[i+1] - values[i]);

    return temp;
}

template<typename __nTp>
std::vector<__nTp> generateDiffEstimate(__nTp slope, __nTp constant, int N) {
    std::vector<__nTp> temp;
    for(int i = 0; i < N; i++)
        temp.push_back(slope*i+constant);
    return temp;
}

template<typename __nTp>
std::vector<__nTp> vectorAppendCumSum(__nTp start, std::vector<__nTp> values) {
    std::vector<__nTp> temp;
    __nTp sum = start;
    temp.push_back(start);
    for (int i = 0; i < values.size(); i++){
        sum += values[i];
        temp.push_back(sum);
    }

    return temp;
}

template<typename __nTp>
void bestFit(__nTp* coeffs, std::vector<__nTp> values) {
    __nTp sumX = 0.0;
    __nTp sumY = 0.0;
    __nTp N = values.size();
    __nTp sumXY = 0.0;
    __nTp sumXX = 0.0;

    for (int i =0; i < values.size(); i++) {
        sumX += (__nTp)i;
        sumY += values[i];
        sumXY += ((__nTp)i*values[i]);
        sumXX += ((__nTp)i * (__nTp)i);
    }

    __nTp numS = N * sumXY - sumX * sumY;
    __nTp den = N * sumXX - sumX * sumX;

    __nTp numC = sumY * sumXX - sumX * sumXY;
    
    coeffs[0] = numS/den;
    coeffs[1] = numC/den;

}

// TODO: Need to work on setting up the grid search

template <typename __nTp, typename __nTpParams>
void grid_cuda_focus_SAR_image(const SAR_Aperture<__nTp>& sar_data,
        const SAR_ImageFormationParameters<__nTpParams>& sar_image_params,
        CArray<__nTp>& output_image, std::ofstream* myfile) {

    switch (sar_image_params.algorithm) {
        case SAR_ImageFormationParameters<__nTpParams>::ALGORITHM::BACKPROJECTION:
            std::cout << "Selected backprojection algorithm for focusing." << std::endl;
            //run_bp(sar_data, sar_image_params, output_image);
            break;
        case SAR_ImageFormationParameters<__nTpParams>::ALGORITHM::MATCHED_FILTER:
            std::cout << "Selected matched filtering algorithm for focusing." << std::endl;
            //run_mf(SARData, SARImgParams, output_image);
            //break;
        default:
            std::cout << "focus_SAR_image()::Algorithm requested is not recognized or available." << std::endl;
            return;
    }

    // Display maximum scene size and resolution
    std::cout << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << sar_image_params.max_Wy_m << " m range, "
            << sar_image_params.max_Wx_m << " m cross-range" << std::endl;
    std::cout << "Maximum Resolution:  " << std::fixed << std::setprecision(2) << sar_image_params.slant_rangeResolution << "m range, "
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

    // Calculate range bins for range compression-based algorithms, e.g., backprojection
    RangeBinData<__nTp> range_bin_data;
    range_bin_data.rangeBins.shape.push_back(sar_image_params.N_fft);
    range_bin_data.rangeBins.shape.push_back(1);
    range_bin_data.rangeBins.data.resize(sar_image_params.N_fft);
    __nTp* rangeBins = &range_bin_data.rangeBins.data[0]; //[sar_image_params.N_fft];
    __nTp minRange = range_bin_data.minRange;
    __nTp maxRange = range_bin_data.maxRange;

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

    cuda_res.copyToDevice("range_vec", (void *) &range_bin_data.rangeBins.data[0],
            range_bin_data.rangeBins.data.size() * sizeof (range_bin_data.rangeBins.data[0]));

    std::cout << cuda_res << std::endl;
    int numSamples = sar_data.sampleData.data.size();
    int newSize = pow(2, ceil(log(sar_data.sampleData.data.size()) / log(2)));

    clock_t c0, c1, c2;

    c0 = clock();
    //std::cout << printf("N_fft: %d, numAzimuthSamples: %d, numSamples: %d\n\n",sar_image_params.N_fft, sar_data.numAzimuthSamples, newSize);
    cuifft(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"), sar_image_params.N_fft, sar_data.numAzimuthSamples);
    cufftNormalize_1DBatch(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"), sar_image_params.N_fft, sar_data.numAzimuthSamples);
    cufftShift_1DBatch<cufftComplex>(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"), sar_image_params.N_fft, sar_data.numAzimuthSamples);
    c1 = clock();
    printf("INFO: CUDA FFT kernels took %f ms.\n", (float) (c1 - c0) * 1000 / CLOCKS_PER_SEC);

    __nTp delta_x_m_per_pix = sar_image_params.Wx_m / (sar_image_params.N_x_pix - 1);
    __nTp delta_y_m_per_pix = sar_image_params.Wy_m / (sar_image_params.N_y_pix - 1);
    __nTp left_m = sar_image_params.x0_m - sar_image_params.Wx_m / 2;
    __nTp bottom_m = sar_image_params.y0_m - sar_image_params.Wy_m / 2;

    // Set up and run the kernel
    dim3 dimBlock(cuda_res.blockwidth, cuda_res.blockheight, 1);
    dim3 dimGrid(std::ceil((float) sar_image_params.N_x_pix / cuda_res.blockwidth),
            std::ceil((float) sar_image_params.N_y_pix / cuda_res.blockheight));
    c0 = clock();

#if ZEROCOPY
    /*
        backprojection_loop << <dimGrid, dimBlock>>>(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"),
                sar_data.numAzimuthSamples, sar_image_params.N_y_pix,
                delta_x, delta_y, sar_data.numRangeSamples, 0, 0,
                c__4_delta_freq, cuda_res.getDeviceMemPointer<float>("startF"),
                left, bottom, cuda_res.getDeviceMemPointer<float4>("platform_positions"), 0, 0,
                cuda_res.getDeviceMemPointer<cufftComplex>("output_image"));
     */
    backprojection_loop<__nTp> << <dimGrid, dimBlock>>>(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"),
            sar_data.numRangeSamples, sar_data.numAzimuthSamples,
            delta_x_m_per_pix, delta_y_m_per_pix,
            left_m, bottom_m, minRange, maxRange,
            cuda_res.getDeviceMemPointer<__nTp>("Ant_x"),
            cuda_res.getDeviceMemPointer<__nTp>("Ant_y"),
            cuda_res.getDeviceMemPointer<__nTp>("Ant_z"),
            cuda_res.getDeviceMemPointer<__nTp>("slant_range"),
            cuda_res.getDeviceMemPointer<__nTp>("startF"),
            cuda_res.getDeviceMemPointer<SAR_ImageFormationParameters < __nTpParams >> ("sar_image_params"),
            cuda_res.getDeviceMemPointer<__nTp>("range_vec"),
            cuda_res.getDeviceMemPointer<cufftComplex>("output_image"));
#else
    // LINE FITTING BASED ON PULSE
    float * xCoeffs = new float[2];
    float * yCoeffs = new float[2];
    float * zCoeffs = new float[2];

    std::vector<NumericType> xPossDiff = vectorDiff(sar_data.Ant_x.data);
    std::vector<NumericType> yPossDiff = vectorDiff(sar_data.Ant_y.data);
    std::vector<NumericType> zPossDiff = vectorDiff(sar_data.Ant_z.data);

    bestFit<NumericType>(xCoeffs, xPossDiff);
    bestFit<NumericType>(yCoeffs, yPossDiff);
    bestFit<NumericType>(zCoeffs, zPossDiff);

    printf("X - Slope coeff = %f\n    Const coeff = %f\n",xCoeffs[0],xCoeffs[1]);
    printf("Y - Slope coeff = %f\n    Const coeff = %f\n",yCoeffs[0],yCoeffs[1]);
    printf("Z - Slope coeff = %f\n    Const coeff = %f\n",zCoeffs[0],zCoeffs[1]);

    *myfile << "gt," << xCoeffs[0] << ',' << xCoeffs[1] << ','
                    << yCoeffs[0] << ',' << yCoeffs[1] << ',' 
                    << zCoeffs[0] << ',' << zCoeffs[1] << ',';

    // GET GRID SEARCH RANGE
    grid_precision gridDiff = 1e-4f;
    grid_precision gridN = 11;

    std::vector<grid_precision> start_point = {(grid_precision) xCoeffs[0]-gridDiff, (grid_precision) xCoeffs[1], 
                                               (grid_precision) yCoeffs[0]-gridDiff, (grid_precision) yCoeffs[1],
                                               (grid_precision) zCoeffs[0]-gridDiff, (grid_precision) zCoeffs[1]};
    std::vector<grid_precision> end_point = {(grid_precision) xCoeffs[0]+gridDiff, (grid_precision) xCoeffs[1], 
                                             (grid_precision) yCoeffs[0]+gridDiff, (grid_precision) yCoeffs[1],
                                             (grid_precision) zCoeffs[0]+gridDiff, (grid_precision) zCoeffs[1]};
    std::vector<grid_precision> grid_numSamples = {(grid_precision) gridN, (grid_precision) 1,
                                              (grid_precision) gridN, (grid_precision) 1,
                                              (grid_precision) gridN, (grid_precision) 1};

    CudaGrid<grid_precision, grid_dimension> grid;
    ck(cudaMalloc(&grid.data(), grid.bytesSize()));

    grid.setStartPoint(start_point);
    grid.setEndPoint(end_point);
    grid.setNumSamples(grid_numSamples);
    grid.display("grid");

    grid_precision axis_sample_counts[grid_dimension];
    grid.getAxisSampleCounts(axis_sample_counts);

    CudaTensor<func_precision, grid_dimension> func_values(axis_sample_counts);
    ck(cudaMalloc(&func_values._data, func_values.bytesSize()));

    // first template argument is the error function return type
    // second template argument is the grid point value type
    CudaGridSearcher<func_precision, grid_precision, grid_dimension> gridsearcher(grid, func_values);

    image_err_func_byvalue host_func_byval_ptr;
    // Copy device function pointer for the function having by-value parameters to host side
    cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr,
                         sizeof(image_err_func_byvalue));

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30));

    int numRSamples = sar_data.numRangeSamples, numASamples = sar_data.numAzimuthSamples;
    cufftComplex* data_p = cuda_res.getDeviceMemPointer<cufftComplex>("sampleData");
    __nTp* ax_p = cuda_res.getDeviceMemPointer<__nTp>("Ant_x");
    __nTp* ay_p = cuda_res.getDeviceMemPointer<__nTp>("Ant_y");
    __nTp* az_p = cuda_res.getDeviceMemPointer<__nTp>("Ant_z");
    __nTp* sr_p = cuda_res.getDeviceMemPointer<__nTp>("slant_range");
    __nTp* sf_p = cuda_res.getDeviceMemPointer<__nTp>("startF");
    SAR_ImageFormationParameters<__nTpParams>* sip_p = cuda_res.getDeviceMemPointer<SAR_ImageFormationParameters < __nTpParams >> ("sar_image_params");
    __nTp* rv_p = cuda_res.getDeviceMemPointer<__nTp>("range_vec");
    cufftComplex* oi_p = cuda_res.getDeviceMemPointer<cufftComplex>("output_image");

    grid_precision testHolder[] = {xCoeffs[0], xCoeffs[1], yCoeffs[0], yCoeffs[1], zCoeffs[0], zCoeffs[1]};
    nv_ext::Vec<grid_precision, grid_dimension> testHolderVec(testHolder);

    c1 = clock();
    gridsearcher.search_by_value_stream(host_func_byval_ptr, 50, 451,
    // gridsearcher.search_by_value(host_func_byval_ptr,
            data_p,
            numRSamples, numASamples,
            delta_x_m_per_pix, delta_y_m_per_pix,
            left_m, bottom_m, 
            minRange, maxRange,
            ax_p,
            ay_p,
            az_p,
            sr_p,
            sf_p,
            sip_p,
            rv_p);
    c2 = clock();
    float searchTime = (float) (c2 - c1) * 1000 / CLOCKS_PER_SEC;
    printf("INFO: cuGridSearch took %f ms.\n", searchTime);

    *myfile << "time," << searchTime << ',';

    func_precision min_value;
    int32_t min_value_index1d;
    func_values.find_extrema(min_value, min_value_index1d);
    
    grid_precision minParams[grid_dimension] = {0};
    grid_precision min_grid_point[grid_dimension];
    grid.getGridPoint(min_grid_point, min_value_index1d);
    std::cout << "Minimum found at point p = { ";
    for (int d=0; d < grid_dimension; d++) {
        minParams[d] = min_grid_point[d];
        std::cout << min_grid_point[d] << ((d < grid_dimension -1) ? ", " : " ");
    }
    std::cout << "}" << std::endl;

    *myfile << "found,";
    for(int i = 0; i < grid_dimension; i++)
        *myfile << minParams[i] << ',';

    nv_ext::Vec<grid_precision, grid_dimension> minParamsVec(minParams);
    computeImageKernel<func_precision, grid_precision, grid_dimension, __nTp><<<1,451>>>(minParamsVec,
            data_p,
            numRSamples, numASamples,
            delta_x_m_per_pix, delta_y_m_per_pix,
            left_m, bottom_m, 
            minRange, maxRange,
            ax_p,
            ay_p,
            az_p,
            sr_p,
            sf_p,
            sip_p,
            rv_p,
            oi_p);
#endif
    /* NOTE: COMMENT IF GRID ONLY */
    c1 = clock();
    printf("INFO: CUDA Backprojection kernel launch took %f ms.\n", (float) (c1 - c0) * 1000 / CLOCKS_PER_SEC);
    if (cudaDeviceSynchronize() != cudaSuccess)
        printf("\nERROR: threads did NOT synchronize! DO NOT TRUST RESULTS!\n\n");
    c2 = clock();
    printf("INFO: CUDA Backprojection execution took %f ms.\n", (float) (c2 - c1) * 1000 / CLOCKS_PER_SEC);
    printf("INFO: CUDA Backprojection total time took %f ms.\n", (float) (c2 - c0) * 1000 / CLOCKS_PER_SEC);
    /**/
#if ZEROCOPY
    int num_img_bytes = sizeof (cufftComplex) * sar_image_params.N_x_pix * sar_image_params.N_y_pix;
    std::vector<cufftComplex> image_data(sar_image_params.N_x_pix * sar_image_params.N_y_pix);
    //cuda_res.copyFromDevice("output_image", &output_image[0], num_img_bytes);
    cuda_res.copyFromZero("output_image", image_data.data(), num_img_bytes);
    for (int idx = 0; idx < sar_image_params.N_x_pix * sar_image_params.N_y_pix; idx++) {
        output_image[idx]._M_real = image_data[idx].x;
        output_image[idx]._M_imag = image_data[idx].y;
    }
    //cuda_res.freeHostMemory("range_vec");
    //from_gpu_complex_to_bp_complex_split(cuda_res.out_image, output_image, sar_image_params.N_x_pix * sar_image_params.N_y_pix);
#else
    int num_img_bytes = sizeof (cufftComplex) * sar_image_params.N_x_pix * sar_image_params.N_y_pix;
    std::vector<cufftComplex> image_data(sar_image_params.N_x_pix * sar_image_params.N_y_pix);
    //cuda_res.copyFromDevice("output_image", &output_image[0], num_img_bytes);
    cuda_res.copyFromDevice("output_image", image_data.data(), num_img_bytes);
    for (int idx = 0; idx < sar_image_params.N_x_pix * sar_image_params.N_y_pix; idx++) {
        output_image[idx]._M_real = image_data[idx].x;
        output_image[idx]._M_imag = image_data[idx].y;
    }

#endif
    cuda_res.freeGPUMemory("range_vec");

    delete xCoeffs;
    delete yCoeffs;
    delete zCoeffs;

    ck(cudaFree(grid.data()));
    ck(cudaFree(func_values.data()));

    if (finalize_CUDAResources(sar_data, sar_image_params, cuda_res) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Problem found de-allocating and free resources on the GPU. Exiting..." << std::endl;
        return;
    }
    std::cout << cuda_res << std::endl;
}

int main(int argc, char **argv) {
    ComplexType test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    ComplexType out[8];
    ComplexArrayType data(test, 8);
    std::unordered_map<std::string, matvar_t*> matlab_readvar_map;

    cxxopts::Options options("cpuBackProjection", "UNC Charlotte Machine Vision Lab SAR Back Projection focusing code.");
    cxxopts_integration(options);

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    bool debug = result["debug"].as<bool>();

    initialize_Sandia_SPHRead(matlab_readvar_map);
    initialize_GOTCHA_MATRead(matlab_readvar_map);

    std::string inputfile;
    if (result.count("input")) {
        inputfile = result["input"].as<std::string>();
    } else {
        std::stringstream ss;

        // Sandia SAR DATA FILE LOADING
        int file_idx = 9; // 1-10 for Sandia Rio Grande, 1-9 for Sandia Farms
        std::string fileprefix = Sandia_RioGrande_fileprefix;
        std::string filepostfix = Sandia_RioGrande_filepostfix;
        //        std::string fileprefix = Sandia_Farms_fileprefix;
        //        std::string filepostfix = Sandia_Farms_filepostfix;
        ss << std::setfill('0') << std::setw(2) << file_idx;


        // GOTCHA SAR DATA FILE LOADING
        int azimuth = 1; // 1-360 for all GOTCHA polarities=(HH,VV,HV,VH) and pass=[pass1,...,pass7] 
        //        std::string fileprefix = GOTCHA_fileprefix;
        //        std::string filepostfix = GOTCHA_filepostfix;
        //        ss << std::setfill('0') << std::setw(3) << azimuth;

        inputfile = fileprefix + ss.str() + filepostfix + ".mat";
    }

    std::cout << "Successfully opened MATLAB file " << inputfile << "." << std::endl;

    SAR_Aperture<NumericType> SAR_aperture_data;
    if (read_MAT_Variables(inputfile, matlab_readvar_map, SAR_aperture_data) == EXIT_FAILURE) {
        std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
        return EXIT_FAILURE;
    }
    // Print out raw data imported from file
    std::cout << SAR_aperture_data << std::endl;

    // Sandia SAR data is multi-channel having up to 4 polarities
    // 1 = HH, 2 = HV, 3 = VH, 4 = VVbandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
    std::string polarity = result["polarity"].as<std::string>();
    if ((polarity == "HH" || polarity == "any") && SAR_aperture_data.sampleData.shape.size() >= 1) {
        SAR_aperture_data.polarity_channel = 0;
    } else if (polarity == "HV" && SAR_aperture_data.sampleData.shape.size() >= 2) {
        SAR_aperture_data.polarity_channel = 1;
    } else if (polarity == "VH" && SAR_aperture_data.sampleData.shape.size() >= 3) {
        SAR_aperture_data.polarity_channel = 2;
    } else if (polarity == "VV" && SAR_aperture_data.sampleData.shape.size() >= 4) {
        SAR_aperture_data.polarity_channel = 3;
    } else {
        std::cout << "Requested polarity channel " << polarity << " is not available." << std::endl;
        return EXIT_FAILURE;
    }
    if (SAR_aperture_data.sampleData.shape.size() > 2) {
        SAR_aperture_data.format_GOTCHA = false;
        // the dimensional index of the polarity index in the 
        // multi-dimensional array (for Sandia SPH SAR data)
        SAR_aperture_data.polarity_dimension = 2;
    }

    initialize_SAR_Aperture_Data(SAR_aperture_data);

    SAR_ImageFormationParameters<NumericType> SAR_image_params =
            SAR_ImageFormationParameters<NumericType>();

    // to increase the frequency samples to a power of 2
    // SAR_image_params.N_fft = (int) 0x01 << (int) (ceil(log2(SAR_aperture_data.numRangeSamples)));
    SAR_image_params.N_fft = (int)SAR_aperture_data.numRangeSamples;
    //SAR_image_params.N_fft = aperture.numRangeSamples;
    SAR_image_params.N_x_pix = (int)SAR_aperture_data.numAzimuthSamples;
    //SAR_image_params.N_y_pix = image_params.N_fft;
    SAR_image_params.N_y_pix = (int)SAR_aperture_data.numRangeSamples;
    // focus image on target phase center
    // Determine the maximum scene size of the image (m)
    // max down-range/fast-time/y-axis extent of image (m)
    SAR_image_params.max_Wy_m = CLIGHT / (2.0 * SAR_aperture_data.mean_deltaF);
    // max cross-range/fast-time/x-axis extent of image (m)
    SAR_image_params.max_Wx_m = CLIGHT / (2.0 * std::abs(SAR_aperture_data.mean_Ant_deltaAz) * SAR_aperture_data.mean_startF);

    // default view is 100% of the maximum possible view
    SAR_image_params.Wx_m = 1.00 * SAR_image_params.max_Wx_m;
    SAR_image_params.Wy_m = 1.00 * SAR_image_params.max_Wy_m;
    // make reconstructed image equal size in (x,y) dimensions
    SAR_image_params.N_x_pix = (int) ((float) SAR_image_params.Wx_m * SAR_image_params.N_y_pix) / SAR_image_params.Wy_m;
    // Determine the resolution of the image (m)
    SAR_image_params.slant_rangeResolution = CLIGHT / (2.0 * SAR_aperture_data.mean_bandwidth);
    SAR_image_params.ground_rangeResolution = SAR_image_params.slant_rangeResolution / std::sin(SAR_aperture_data.mean_Ant_El);
    SAR_image_params.azimuthResolution = CLIGHT / (2.0 * SAR_aperture_data.Ant_totalAz * SAR_aperture_data.mean_startF);

    // Print out data after critical data fields for SAR focusing have been computed
    std::cout << SAR_aperture_data << std::endl;

    SAR_Aperture<NumericType> SAR_focusing_data;
    if (!SAR_aperture_data.format_GOTCHA) {
        //SAR_aperture_data.exportData(SAR_focusing_data, SAR_aperture_data.polarity_channel);
        SAR_aperture_data.exportData(SAR_focusing_data, 2);
    } else {
        SAR_focusing_data = SAR_aperture_data;
    }

    //    SAR_ImageFormationParameters<NumericType> SAR_image_params =
    //            SAR_ImageFormationParameters<NumericType>::create<NumericType>(SAR_focusing_data);

    std::cout << "Data for focusing" << std::endl;
    std::cout << SAR_focusing_data << std::endl;

    std::ofstream myfile;
    myfile.open("collectedData.txt", std::ios::out | std::ios::app);
    myfile << inputfile.c_str() << ',';

    printf("Main: deltaAz = %f, deltaF = %f, mean_startF = %f\nmaxWx_m = %f, maxWy_m = %f, Wx_m = %f, Wy_m = %f\nX_pix = %d, Y_pix = %d\nNum Az = %d, Num range = %d\n", SAR_aperture_data.mean_Ant_deltaAz, SAR_aperture_data.mean_startF, SAR_aperture_data.mean_deltaF, SAR_image_params.max_Wx_m, SAR_image_params.max_Wy_m, SAR_image_params.Wx_m, SAR_image_params.Wy_m, SAR_image_params.N_x_pix, SAR_image_params.N_y_pix, SAR_aperture_data.numAzimuthSamples, SAR_aperture_data.numRangeSamples);
    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);

    grid_cuda_focus_SAR_image(SAR_focusing_data, SAR_image_params, output_image, &myfile);

    // Required parameters for output generation manually overridden by command line arguments
    std::string output_filename = result["output"].as<std::string>();
    SAR_image_params.dyn_range_dB = result["dynrange"].as<float>();

    writeBMPFile(SAR_image_params, output_image, output_filename);
    myfile << '\n';
    myfile.close();
    return EXIT_SUCCESS;
}
