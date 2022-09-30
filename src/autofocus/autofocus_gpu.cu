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

#include <third_party/log.h>
#include <third_party/cxxopts.hpp>

#include <cuGridSearch.cuh>

// this declaration needs to be in any C++ compiled target for CPU
//#define CUDAFUNCTION

#include <charlotte_sar_api.hpp>
#include <uncc_sar_globals.hpp>

typedef float pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

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

int main(int argc, char **argv) {
    cxxopts::Options options("example_gpu", "UNC Charlotte Machine Vision Lab SAR Back Projection focusing code.");
    cxxopts_integration(options);

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    bool debug = result["debug"].as<bool>();

    if (result.count("output")) {
        outputfile = result["output"].as<std::string>();
    } else {
        outputfile = "output_gpu.bmp";
    }

    std::string inputfile;
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
        int azimuth = 1; // 1-360 for all GOTCHA polarities=(HH,VV,HV,VH) and pass=[pass1,...,pass7] 
        //        std::string fileprefix = GOTCHA_fileprefix;
        //        std::string filepostfix = GOTCHA_filepostfix;
        //        ss << std::setfill('0') << std::setw(3) << azimuth;

        inputfile = fileprefix + ss.str() + filepostfix + ".mat";
    }

    // Sandia SAR data is multi-channel having up to 4 polarities
    // 1 = HH, 2 = HV, 3 = VH, 4 = VV bandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
    std::string polarity = result["polarity"].as<std::string>();

    if (verbose)
        std::cout << "Successfully opened MATLAB file " << inputfile << "." << std::endl;

    PhaseHistory<float> ph;
    ph.id = 0;

    int MAX_PULSES = 200;
    readData(inputfile, MAX_PULSES, polarity, ph, verbose);

    CudaImage<pixel_precision> m1(6, 6);
    CudaImage<pixel_precision> m2(6, 6);

    checkCudaErrors(cudaMalloc(&m1.data(), m1.bytesSize()));
    checkCudaErrors(cudaMalloc(&m2.data(), m2.bytesSize()));

    m1.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));
    m2.setValuesFromVector(std::vector<pixel_precision>(imageB_data, imageB_data + 6 * 6));

    initLogger(("logfile_" + DateTime() + ".log").c_str(), ldebug);

    sph_sar_data_callback_gpu<float>(
            ph.id,
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

    endLogger();

    return EXIT_SUCCESS;
}
