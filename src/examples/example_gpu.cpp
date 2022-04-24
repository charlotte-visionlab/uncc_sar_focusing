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
#include <stdio.h>  /* printf */
#include <time.h>

#include <iomanip>
#include <sstream>

#include <cxxopts.hpp>

// this declaration needs to be in any C++ compiled target for CPU
//#define CUDAFUNCTION __host__ __device__
#define CUDAFUNCTION

#include "charlotte_sar_api.hpp"
#include "example_gpu.hpp"

#include <uncc_sar_globals.hpp>
//#include "../gpuBackProjection/cuda_sar_focusing/cuda_sar_focusing.hpp"

//using NumericType = float;
//using ComplexType = Complex<NumericType>;
//using ComplexArrayType = CArray<NumericType>;

int main(int argc, char **argv) {
    cxxopts::Options options("example_gpu", "UNC Charlotte Machine Vision Lab SAR Back Projection focusing code.");
    cxxopts_integration(options);

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    bool debug = result["debug"].as<bool>();

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

    // Sandia SAR data is multi-channel having up to 4 polarities
    // 1 = HH, 2 = HV, 3 = VH, 4 = VVbandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
    std::string polarity = result["polarity"].as<std::string>();

    std::cout << "Successfully opened MATLAB file " << inputfile << "." << std::endl;

    PhaseHistory<float> ph;

    readData(inputfile, polarity, ph);

    sph_sar_data_callback_gpu<float>(
            ph.sph_MATData_preamble_ADF,
            ph.sph_MATData_Data_SampleData.data(),
            ph.numRangeSamples,
            ph.sph_MATData_Data_StartF.data(),
            ph.sph_MATData_Data_ChirpRate.data(),
            ph.sph_MATData_Data_radarCoordinateFrame_x.data(),
            ph.sph_MATData_Data_radarCoordinateFrame_y.data(),
            ph.sph_MATData_Data_radarCoordinateFrame_z.data(),
            ph.numAzimuthSamples
            );

    return EXIT_SUCCESS;
    //    ComplexType test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    //    ComplexType out[8];
    //    ComplexArrayType data(test, 8);
    //    std::unordered_map<std::string, matvar_t*> matlab_readvar_map;
    //
    //    initialize_Sandia_SPHRead(matlab_readvar_map);
    //    initialize_GOTCHA_MATRead(matlab_readvar_map);
    //
    //    SAR_Aperture<NumericType> SAR_aperture_data;
    //    if (read_MAT_Variables(inputfile, matlab_readvar_map, SAR_aperture_data) == EXIT_FAILURE) {
    //        std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
    //        return EXIT_FAILURE;
    //    }
    //    // Print out raw data imported from file
    //    std::cout << SAR_aperture_data << std::endl;
    //
    //    // Sandia SAR data is multi-channel having up to 4 polarities
    //    // 1 = HH, 2 = HV, 3 = VH, 4 = VVbandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
    //    std::string polarity = result["polarity"].as<std::string>();
    //    if ((polarity == "HH" || polarity == "any") && SAR_aperture_data.sampleData.shape.size() >= 1) {
    //        SAR_aperture_data.polarity_channel = 0;
    //    } else if (polarity == "HV" && SAR_aperture_data.sampleData.shape.size() >= 2) {
    //        SAR_aperture_data.polarity_channel = 1;
    //    } else if (polarity == "VH" && SAR_aperture_data.sampleData.shape.size() >= 3) {
    //        SAR_aperture_data.polarity_channel = 2;
    //    } else if (polarity == "VV" && SAR_aperture_data.sampleData.shape.size() >= 4) {
    //        SAR_aperture_data.polarity_channel = 3;
    //    } else {
    //        std::cout << "Requested polarity channel " << polarity << " is not available." << std::endl;
    //        return EXIT_FAILURE;
    //    }
    //    if (SAR_aperture_data.sampleData.shape.size() > 2) {
    //        SAR_aperture_data.format_GOTCHA = false;
    //        // the dimensional index of the polarity index in the 
    //        // multi-dimensional array (for Sandia SPH SAR data)
    //        SAR_aperture_data.polarity_dimension = 2;
    //    }
    //
    //    SAR_ImageFormationParameters<NumericType> SAR_image_params =
    //            SAR_ImageFormationParameters<NumericType>();
    //
    //    // to increase the frequency samples to a power of 2
    //    SAR_image_params.N_fft = (int) 0x01 << (int) (ceil(log2(SAR_aperture_data.numRangeSamples)));
    //    //SAR_image_params.N_fft = aperture.numRangeSamples;
    //    SAR_image_params.N_x_pix = SAR_aperture_data.numAzimuthSamples;
    //    //SAR_image_params.N_y_pix = image_params.N_fft;
    //    SAR_image_params.N_y_pix = SAR_aperture_data.numRangeSamples;
    //    // focus image on target phase center
    //    // Determine the maximum scene size of the image (m)
    //    // max down-range/fast-time/y-axis extent of image (m)
    //    SAR_image_params.max_Wy_m = CLIGHT / (2.0 * SAR_aperture_data.mean_deltaF);
    //    // max cross-range/fast-time/x-axis extent of image (m)
    //    SAR_image_params.max_Wx_m = CLIGHT / (2.0 * std::abs(SAR_aperture_data.mean_Ant_deltaAz) * SAR_aperture_data.mean_startF);
    //
    //    // default view is 100% of the maximum possible view
    //    SAR_image_params.Wx_m = 1.00 * SAR_image_params.max_Wx_m;
    //    SAR_image_params.Wy_m = 1.00 * SAR_image_params.max_Wy_m;
    //    // make reconstructed image equal size in (x,y) dimensions
    //    SAR_image_params.N_x_pix = (int) ((float) SAR_image_params.Wx_m * SAR_image_params.N_y_pix) / SAR_image_params.Wy_m;
    //    // Determine the resolution of the image (m)
    //    SAR_image_params.slant_rangeResolution = CLIGHT / (2.0 * SAR_aperture_data.mean_bandwidth);
    //    SAR_image_params.ground_rangeResolution = SAR_image_params.slant_rangeResolution / std::sin(SAR_aperture_data.mean_Ant_El);
    //    SAR_image_params.azimuthResolution = CLIGHT / (2.0 * SAR_aperture_data.Ant_totalAz * SAR_aperture_data.mean_startF);
    //
    //    initialize_SAR_Aperture_Data(SAR_aperture_data);
    //    // Print out data after critical data fields for SAR focusing have been computed
    //    std::cout << SAR_aperture_data << std::endl;
    //
    //    SAR_Aperture<NumericType> SAR_focusing_data;
    //    if (!SAR_aperture_data.format_GOTCHA) {
    //        //SAR_aperture_data.exportData(SAR_focusing_data, SAR_aperture_data.polarity_channel);
    //        SAR_aperture_data.exportData(SAR_focusing_data, 2);
    //    } else {
    //        SAR_focusing_data = SAR_aperture_data;
    //    }
    //
    //    //    SAR_ImageFormationParameters<NumericType> SAR_image_params =
    //    //            SAR_ImageFormationParameters<NumericType>::create<NumericType>(SAR_focusing_data);
    //
    //    std::cout << "Data for focusing" << std::endl;
    //    std::cout << SAR_focusing_data << std::endl;
    //
    //    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);
    //
    //    cuda_focus_SAR_image(SAR_focusing_data, SAR_image_params, output_image);
    //
    //    // Required parameters for output generation manually overridden by command line arguments
    //    std::string output_filename = result["output"].as<std::string>();
    //    SAR_image_params.dyn_range_dB = result["dynrange"].as<float>();
    //
    //    writeBMPFile(SAR_image_params, output_image, output_filename);
    return EXIT_SUCCESS;
}

//int main(int argc, char **argv) {
//    cxxopts::Options options("cpuBackProjection", "UNC Charlotte Machine Vision Lab SAR Back Projection focusing code.");
//    cxxopts_integration(options);
//
//    auto result = options.parse(argc, argv);
//
//    if (result.count("help")) {
//        std::cout << options.help() << std::endl;
//        exit(0);
//    }
//    bool debug = result["debug"].as<bool>();
//
//    std::string inputfile;
//    if (result.count("input")) {
//        inputfile = result["input"].as<std::string>();
//    } else {
//        std::stringstream ss;
//
//        // Sandia SAR DATA FILE LOADING
//        int file_idx = 9; // 1-10 for Sandia Rio Grande, 1-9 for Sandia Farms
//        std::string fileprefix = Sandia_RioGrande_fileprefix;
//        std::string filepostfix = Sandia_RioGrande_filepostfix;
//        //        std::string fileprefix = Sandia_Farms_fileprefix;
//        //        std::string filepostfix = Sandia_Farms_filepostfix;
//        ss << std::setfill('0') << std::setw(2) << file_idx;
//
//
//        // GOTCHA SAR DATA FILE LOADING
//        int azimuth = 1; // 1-360 for all GOTCHA polarities=(HH,VV,HV,VH) and pass=[pass1,...,pass7] 
//        //        std::string fileprefix = GOTCHA_fileprefix;
//        //        std::string filepostfix = GOTCHA_filepostfix;
//        //        ss << std::setfill('0') << std::setw(3) << azimuth;
//
//        inputfile = fileprefix + ss.str() + filepostfix + ".mat";
//    }
//
//    std::cout << "Successfully opened MATLAB file " << inputfile << "." << std::endl;
//
//    ComplexType test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
//    ComplexType out[8];
//    ComplexArrayType data(test, 8);
//    std::unordered_map<std::string, matvar_t*> matlab_readvar_map;
//
//    initialize_Sandia_SPHRead(matlab_readvar_map);
//    initialize_GOTCHA_MATRead(matlab_readvar_map);
//
//    SAR_Aperture<NumericType> SAR_aperture_data;
//    if (read_MAT_Variables(inputfile, matlab_readvar_map, SAR_aperture_data) == EXIT_FAILURE) {
//        std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
//        return EXIT_FAILURE;
//    }
//    // Print out raw data imported from file
//    std::cout << SAR_aperture_data << std::endl;
//
//    // Sandia SAR data is multi-channel having up to 4 polarities
//    // 1 = HH, 2 = HV, 3 = VH, 4 = VVbandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
//    std::string polarity = result["polarity"].as<std::string>();
//    if ((polarity == "HH" || polarity == "any") && SAR_aperture_data.sampleData.shape.size() >= 1) {
//        SAR_aperture_data.polarity_channel = 0;
//    } else if (polarity == "HV" && SAR_aperture_data.sampleData.shape.size() >= 2) {
//        SAR_aperture_data.polarity_channel = 1;
//    } else if (polarity == "VH" && SAR_aperture_data.sampleData.shape.size() >= 3) {
//        SAR_aperture_data.polarity_channel = 2;
//    } else if (polarity == "VV" && SAR_aperture_data.sampleData.shape.size() >= 4) {
//        SAR_aperture_data.polarity_channel = 3;
//    } else {
//        std::cout << "Requested polarity channel " << polarity << " is not available." << std::endl;
//        return EXIT_FAILURE;
//    }
//    if (SAR_aperture_data.sampleData.shape.size() > 2) {
//        SAR_aperture_data.format_GOTCHA = false;
//        // the dimensional index of the polarity index in the 
//        // multi-dimensional array (for Sandia SPH SAR data)
//        SAR_aperture_data.polarity_dimension = 2;
//    }
//
//    SAR_ImageFormationParameters<NumericType> SAR_image_params =
//            SAR_ImageFormationParameters<NumericType>();
//
//    // to increase the frequency samples to a power of 2
//    SAR_image_params.N_fft = (int) 0x01 << (int) (ceil(log2(SAR_aperture_data.numRangeSamples)));
//    //SAR_image_params.N_fft = aperture.numRangeSamples;
//    SAR_image_params.N_x_pix = SAR_aperture_data.numAzimuthSamples;
//    //SAR_image_params.N_y_pix = image_params.N_fft;
//    SAR_image_params.N_y_pix = SAR_aperture_data.numRangeSamples;
//    // focus image on target phase center
//    // Determine the maximum scene size of the image (m)
//    // max down-range/fast-time/y-axis extent of image (m)
//    SAR_image_params.max_Wy_m = CLIGHT / (2.0 * SAR_aperture_data.mean_deltaF);
//    // max cross-range/fast-time/x-axis extent of image (m)
//    SAR_image_params.max_Wx_m = CLIGHT / (2.0 * std::abs(SAR_aperture_data.mean_Ant_deltaAz) * SAR_aperture_data.mean_startF);
//
//    // default view is 100% of the maximum possible view
//    SAR_image_params.Wx_m = 1.00 * SAR_image_params.max_Wx_m;
//    SAR_image_params.Wy_m = 1.00 * SAR_image_params.max_Wy_m;
//    // make reconstructed image equal size in (x,y) dimensions
//    SAR_image_params.N_x_pix = (int) ((float) SAR_image_params.Wx_m * SAR_image_params.N_y_pix) / SAR_image_params.Wy_m;
//    // Determine the resolution of the image (m)
//    SAR_image_params.slant_rangeResolution = CLIGHT / (2.0 * SAR_aperture_data.mean_bandwidth);
//    SAR_image_params.ground_rangeResolution = SAR_image_params.slant_rangeResolution / std::sin(SAR_aperture_data.mean_Ant_El);
//    SAR_image_params.azimuthResolution = CLIGHT / (2.0 * SAR_aperture_data.Ant_totalAz * SAR_aperture_data.mean_startF);
//
//    initialize_SAR_Aperture_Data(SAR_aperture_data);
//    // Print out data after critical data fields for SAR focusing have been computed
//    std::cout << SAR_aperture_data << std::endl;
//
//    SAR_Aperture<NumericType> SAR_focusing_data;
//    if (!SAR_aperture_data.format_GOTCHA) {
//        //SAR_aperture_data.exportData(SAR_focusing_data, SAR_aperture_data.polarity_channel);
//        SAR_aperture_data.exportData(SAR_focusing_data, 2);
//    } else {
//        SAR_focusing_data = SAR_aperture_data;
//    }
//
//    //    SAR_ImageFormationParameters<NumericType> SAR_image_params =
//    //            SAR_ImageFormationParameters<NumericType>::create<NumericType>(SAR_focusing_data);
//
//    std::cout << "Data for focusing" << std::endl;
//    std::cout << SAR_focusing_data << std::endl;
//
//    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);
//
//    cuda_focus_SAR_image(SAR_focusing_data, SAR_image_params, output_image);
//
//    // Required parameters for output generation manually overridden by command line arguments
//    std::string output_filename = result["output"].as<std::string>();
//    SAR_image_params.dyn_range_dB = result["dynrange"].as<float>();
//
//    writeBMPFile(SAR_image_params, output_image, output_filename);
//    return EXIT_SUCCESS;
//}
