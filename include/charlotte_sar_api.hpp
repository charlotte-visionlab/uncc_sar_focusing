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

/* 
 * File:   charlotte_sar_api.hpp
 * Author: local-arwillis
 *
 * Created on April 7, 2022, 10:53 AM
 */

#ifndef EXTERNAL_API_HPP
#define EXTERNAL_API_HPP

#include <vector>
#include <string>

template<typename __nTp>
class PhaseHistory {
public:
    __nTp sph_MATData_preamble_ADF;
    std::vector<__nTp> sph_MATData_Data_SampleData;
    int numRangeSamples;
    std::vector<__nTp> sph_MATData_Data_StartF;
    std::vector<__nTp> sph_MATData_Data_ChirpRate;
    std::vector<__nTp> sph_MATData_Data_radarCoordinateFrame_x;
    std::vector<__nTp> sph_MATData_Data_radarCoordinateFrame_y;
    std::vector<__nTp> sph_MATData_Data_radarCoordinateFrame_z;
    int numAzimuthSamples;
};

extern int readData(const std::string& inputfile, const std::string& polarity, PhaseHistory<float>& ph);

template<typename __nTp>
int sph_sar_data_callback_cpu(
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_SampleData,
        int numRangeSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numAzimuthSamples);

template<typename __nTp>
int sph_sar_data_callback_gpu(
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_SampleData,
        int numRangeSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numAzimuthSamples);
//template<typename __nTp>
//int sph_sar_data_callback_cpu(
//        __nTp sph_MATData_preamble_ADF,
//        __nTp *sph_MATData_Data_SampleData,
//        int numRangeSamples,
//        __nTp *sph_MATData_Data_StartF,
//        __nTp *sph_MATData_Data_ChirpRate,
//        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
//        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
//        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
//        int numAzimuthSamples) {
//    return EXIT_SUCCESS;
//}

//template<typename __nTp>
//int sph_sar_data_callback_gpu(
//        __nTp sph_MATData_preamble_ADF,
//        __nTp *sph_MATData_Data_SampleData,
//        int numRangeSamples,
//        __nTp *sph_MATData_Data_StartF,
//        __nTp *sph_MATData_Data_ChirpRate,
//        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
//        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
//        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
//        int numAzimuthSamples) {
//
//    SAR_Aperture<__nTp> SAR_focusing_data;
//
//    SAR_focusing_data.ADF.shape.push_back(1);
//    SAR_focusing_data.ADF.data.push_back(sph_MATData_preamble_ADF);
//
//    int numPulseVec[1] = {numAzimuthSamples};
//    //import_MatrixComplex<__nTp, __nTp>(sph_MATData_Data_SampleData, int[]{numSamples, numPulses}, 2, SAR_focusing_data.sampleData);
//    import_Vector<__nTp, __nTp>(sph_MATData_Data_StartF, numPulseVec, 1, SAR_focusing_data.startF);
//    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_x, numPulseVec, 1, SAR_focusing_data.Ant_x);
//    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_y, numPulseVec, 1, SAR_focusing_data.Ant_y);
//    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_z, numPulseVec, 1, SAR_focusing_data.Ant_z);
//
//    // convert chirp rate to deltaF
//    std::vector<__nTp> deltaF(numAzimuthSamples);
//    for (int i = 0; i < numAzimuthSamples; i++) {
//        deltaF[i] = sph_MATData_Data_ChirpRate[i] / sph_MATData_preamble_ADF;
//    }
//    import_Vector<__nTp, __nTp>(sph_MATData_Data_StartF, numPulseVec, 1, SAR_focusing_data.startF);
//
//    SAR_ImageFormationParameters<__nTp> SAR_image_params; // =
//    //SAR_ImageFormationParameters<__nTp>::create<__nTp>(SAR_focusing_data);
//
//    std::cout << SAR_image_params << std::endl;
//
//    //ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);
//    // Required parameters for output generation manually overridden by command line arguments
//    //std::string output_filename = result["output"].as<std::string>();
//    //SAR_image_params.dyn_range_dB = result["dynrange"].as<float>();
//
//    //writeBMPFile(SAR_image_params, output_image, output_filename);
//
//    return EXIT_SUCCESS;
//}

template<typename __nTp>
int sar_data_callback(
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_SampleData,
        int numRangeSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numAzimuthSamples) {

    sph_sar_data_callback_cpu<__nTp>(
            sph_MATData_preamble_ADF,
            sph_MATData_Data_SampleData,
            numRangeSamples,
            sph_MATData_Data_StartF,
            sph_MATData_Data_ChirpRate,
            sph_MATData_Data_radarCoordinateFrame_x,
            sph_MATData_Data_radarCoordinateFrame_y,
            sph_MATData_Data_radarCoordinateFrame_z,
            numAzimuthSamples);

//    sph_sar_data_callback_gpu<__nTp>(
//            sph_MATData_preamble_ADF,
//            sph_MATData_Data_SampleData,
//            numRangeSamples,
//            sph_MATData_Data_StartF,
//            sph_MATData_Data_ChirpRate,
//            sph_MATData_Data_radarCoordinateFrame_x,
//            sph_MATData_Data_radarCoordinateFrame_y,
//            sph_MATData_Data_radarCoordinateFrame_z,
//            numAzimuthSamples);

    return EXIT_SUCCESS;
}

#endif /* EXTERNAL_API_HPP */

