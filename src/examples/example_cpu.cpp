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

#include <iomanip>
#include <sstream>

#include <cxxopts.hpp>

// this declaration needs to be in any C++ compiled target for CPU
#define CUDAFUNCTION

#include "charlotte_sar_api.hpp"

#include "../cpuBackProjection/uncc_sar_focusing.hpp"
#include "../cpuBackProjection/cpuBackProjection.hpp"
#include "../cpuBackProjection/cpuBackProjection_main.hpp"

using NumericType = float;
using ComplexType = Complex<NumericType>;
using ComplexArrayType = CArray<NumericType>;

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
    std::string inputfile;

    initialize_Sandia_SPHRead(matlab_readvar_map);
    initialize_GOTCHA_MATRead(matlab_readvar_map);

    if (result.count("input")) {
        inputfile = result["input"].as<std::string>();
    } else {
        std::stringstream ss;

        // Sandia SAR DATA FILE LOADING
        int file_idx = 1; // 1-10 for Sandia Rio Grande, 1-9 for Sandia Farms
        std::string fileprefix = Sandia_RioGrande_fileprefix;
        std::string filepostfix = Sandia_RioGrande_filepostfix;
        //        std::string fileprefix = Sandia_Farms_fileprefix;
        //        std::string filepostfix = Sandia_Farms_filepostfix;
        ss << std::setfill('0') << std::setw(2) << file_idx;

        // GOTCHA SAR DATA FILE LOADING
        //int azimuth = 1; // 1-360 for all GOTCHA polarities=(HH,VV,HV,VH) and pass=[pass1,...,pass7] 
        //std::string fileprefix = GOTCHA_fileprefix;
        //std::string filepostfix = GOTCHA_filepostfix;
        //ss << std::setfill('0') << std::setw(3) << azimuth;

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
    }
    if (!SAR_aperture_data.format_GOTCHA) {
        // the dimensional index of the polarity index in the 
        // multi-dimensional array (for Sandia SPH SAR data)
        SAR_aperture_data.polarity_dimension = 2;
    }
    // Print out data after critical data fields for SAR focusing have been computed
    initialize_SAR_Aperture_Data(SAR_aperture_data);

    std::cout << SAR_aperture_data << std::endl;

    SAR_ImageFormationParameters<NumericType> SAR_image_params =
            SAR_ImageFormationParameters<NumericType>::create<NumericType>(SAR_aperture_data);

    std::cout << SAR_image_params << std::endl;

    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);

    // Allocate memory for data to pass to library function call
    NumericType sph_MATData_preamble_ADF;
    NumericType sph_MATData_Data_SampleData[2 * SAR_aperture_data.numRangeSamples * SAR_aperture_data.numAzimuthSamples];
    NumericType sph_MATData_Data_StartF[SAR_aperture_data.numAzimuthSamples];
    NumericType sph_MATData_Data_ChirpRate[SAR_aperture_data.numAzimuthSamples];
    NumericType sph_MATData_Data_radarCoordinateFrame_x[SAR_aperture_data.numAzimuthSamples];
    NumericType sph_MATData_Data_radarCoordinateFrame_y[SAR_aperture_data.numAzimuthSamples];
    NumericType sph_MATData_Data_radarCoordinateFrame_z[SAR_aperture_data.numAzimuthSamples];
    
    //std::vector<std::vector<int>> adf_index {{0}};
    //sph_MATData_preamble_ADF = (NumericType) SAR_aperture_data.ADF.getData(adf_index).at(0);
    std::vector<NumericType> vec_ADF = SAR_aperture_data.ADF.toVector<NumericType>();
    sph_MATData_preamble_ADF = vec_ADF[0];

    std::vector<ComplexType> vec_SampleData = SAR_aperture_data.sampleData.toVector();
    //sph_MATData_Data_SampleData
            
    std::vector<std::vector<int>> pulse_indices;    
    std::vector<int> pulse_index_values;
    for (int i=0; i < SAR_aperture_data.numAzimuthSamples; i++) {        
        pulse_index_values.push_back(i);
    }
    pulse_indices.push_back(pulse_index_values);
    //    std::vector<NumericType> vec_StartF = SAR_aperture_data.startF.getData(pulse_indices);
    //    std::vector<NumericType> vec_ChirpRate = SAR_aperture_data.chirpRate.getData(pulse_indices);
    //    std::vector<NumericType> vec_radarCoordinateFrame_x = SAR_aperture_data.Ant_x.getData(pulse_indices);
    //    std::vector<NumericType> vec_radarCoordinateFrame_y = SAR_aperture_data.Ant_y.getData(pulse_indices);
    //    std::vector<NumericType> vec_radarCoordinateFrame_z = SAR_aperture_data.Ant_z.getData(pulse_indices);
    std::vector<NumericType> vec_StartF = SAR_aperture_data.startF.toVector<NumericType>();
    std::vector<NumericType> vec_ChirpRate = SAR_aperture_data.chirpRate.toVector<NumericType>();
    std::vector<NumericType> vec_radarCoordinateFrame_x = SAR_aperture_data.Ant_x.toVector<NumericType>();
    std::vector<NumericType> vec_radarCoordinateFrame_y = SAR_aperture_data.Ant_y.toVector<NumericType>();
    std::vector<NumericType> vec_radarCoordinateFrame_z = SAR_aperture_data.Ant_z.toVector<NumericType>();
    for (int i=0; i < SAR_aperture_data.numAzimuthSamples; i++) {        
        sph_MATData_Data_StartF[i] = vec_StartF[i];
        sph_MATData_Data_ChirpRate[i] = vec_ChirpRate[i];
        sph_MATData_Data_radarCoordinateFrame_x[i] = vec_radarCoordinateFrame_x[i];
        sph_MATData_Data_radarCoordinateFrame_y[i] = vec_radarCoordinateFrame_y[i];
        sph_MATData_Data_radarCoordinateFrame_z[i] = vec_radarCoordinateFrame_z[i];
    }
        
    //focus_SAR_image(SAR_aperture_data, SAR_image_params, output_image);
    sar_data_callback<NumericType>(
            SAR_aperture_data.numAzimuthSamples,    
            sph_MATData_preamble_ADF,
            sph_MATData_Data_SampleData,
            SAR_aperture_data.numRangeSamples,
            sph_MATData_Data_StartF,
            sph_MATData_Data_ChirpRate,
            sph_MATData_Data_radarCoordinateFrame_x,
            sph_MATData_Data_radarCoordinateFrame_y,
            sph_MATData_Data_radarCoordinateFrame_z,
            SAR_aperture_data.numAzimuthSamples
            );

    // Required parameters for output generation manually overridden by command line arguments
    std::string output_filename = result["output"].as<std::string>();
    SAR_image_params.dyn_range_dB = result["dynrange"].as<float>();

    writeBMPFile(SAR_image_params, output_image, output_filename);
    return EXIT_SUCCESS;
}
