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
 * File:   example_gpu.hpp
 * Author: local-arwillis
 *
 * Created on April 6, 2022, 11:05 AM
 */

#ifndef EXAMPLE_GPU_HPP
#define EXAMPLE_GPU_HPP

#include <cstdlib>

#include <valarray>
#include <uncc_complex.hpp>

#include <uncc_sar_focusing.hpp>
#include <uncc_sar_matio.hpp>

using NumericType = float;
using ComplexType = Complex<NumericType>;
using ComplexArrayType = CArray<NumericType>;

template <typename __nTp, typename __nTpParams>
void cuda_focus_SAR_image(const SAR_Aperture<__nTp>& sar_data,
        const SAR_ImageFormationParameters<__nTpParams>& sar_image_params,
        CArray<__nTp>& output_image);
//template void cuda_focus_SAR_image<float, float>(SAR_Aperture<float> const&, SAR_ImageFormationParameters<float> const&, std::valarray<unccComplex<float> >&);

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
        int numAzimuthSamples) {
    SAR_Aperture<__nTp> SAR_focusing_data;

    SAR_focusing_data.ADF.shape.push_back(1);
    SAR_focusing_data.ADF.data.push_back(sph_MATData_preamble_ADF);

    int numPulseVec[1] = {numAzimuthSamples};
    int phaseHistoryDims[2] = {numRangeSamples, numAzimuthSamples};
    import_MatrixComplex<__nTp, Complex<__nTp> >(sph_MATData_Data_SampleData, phaseHistoryDims, 2, SAR_focusing_data.sampleData);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_StartF, numPulseVec, 1, SAR_focusing_data.startF);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_ChirpRate, numPulseVec, 1, SAR_focusing_data.chirpRate);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_x, numPulseVec, 1, SAR_focusing_data.Ant_x);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_y, numPulseVec, 1, SAR_focusing_data.Ant_y);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_z, numPulseVec, 1, SAR_focusing_data.Ant_z);

    // convert chirp rate to deltaF
    //    std::vector<__nTp> deltaF(numAzimuthSamples);
    //    for (int i = 0; i < numAzimuthSamples; i++) {
    //        deltaF[i] = sph_MATData_Data_ChirpRate[i] / sph_MATData_preamble_ADF;
    //    }
    //    import_Vector<__nTp, __nTp>(sph_MATData_Data_StartF, numPulseVec, 1, SAR_focusing_data.startF);

    SAR_focusing_data.format_GOTCHA = false;
    initialize_SAR_Aperture_Data(SAR_focusing_data);

    SAR_ImageFormationParameters<__nTp> SAR_image_params; // =
    //SAR_ImageFormationParameters<__nTp>::create<__nTp>(SAR_focusing_data);

    // to increase the frequency samples to a power of 2
    //SAR_image_params.N_fft = (int) 0x01 << (int) (ceil(log2(SAR_aperture_data.numRangeSamples)));
    SAR_image_params.N_fft = numRangeSamples;
    SAR_image_params.N_x_pix = numAzimuthSamples;
    //SAR_image_params.N_y_pix = image_params.N_fft;
    SAR_image_params.N_y_pix = numRangeSamples;
    // focus image on target phase center
    // Determine the maximum scene size of the image (m)
    // max down-range/fast-time/y-axis extent of image (m)
    SAR_image_params.max_Wy_m = CLIGHT / (2.0 * SAR_focusing_data.mean_deltaF);
    // max cross-range/fast-time/x-axis extent of image (m)
    SAR_image_params.max_Wx_m = CLIGHT / (2.0 * std::abs(SAR_focusing_data.mean_Ant_deltaAz) * SAR_focusing_data.mean_startF);

    // default view is 100% of the maximum possible view
    SAR_image_params.Wx_m = 1.00 * SAR_image_params.max_Wx_m;
    SAR_image_params.Wy_m = 1.00 * SAR_image_params.max_Wy_m;
    // make reconstructed image equal size in (x,y) dimensions
    SAR_image_params.N_x_pix = (int) ((float) SAR_image_params.Wx_m * SAR_image_params.N_y_pix) / SAR_image_params.Wy_m;
    // Determine the resolution of the image (m)
    SAR_image_params.slant_rangeResolution = CLIGHT / (2.0 * SAR_focusing_data.mean_bandwidth);
    SAR_image_params.ground_rangeResolution = SAR_image_params.slant_rangeResolution / std::sin(SAR_focusing_data.mean_Ant_El);
    SAR_image_params.azimuthResolution = CLIGHT / (2.0 * SAR_focusing_data.Ant_totalAz * SAR_focusing_data.mean_startF);

    std::cout << SAR_image_params << std::endl;
    std::cout << "Data for focusing" << std::endl;
    std::cout << SAR_focusing_data << std::endl;

    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);

    cuda_focus_SAR_image(SAR_focusing_data, SAR_image_params, output_image);

    // Required parameters for output generation manually overridden by command line arguments
    std::string output_filename = "output_cpu.bmp";
    SAR_image_params.dyn_range_dB = 70;

    writeBMPFile(SAR_image_params, output_image, output_filename);

    return EXIT_SUCCESS;
}

#endif /* EXAMPLE_GPU_HPP */

