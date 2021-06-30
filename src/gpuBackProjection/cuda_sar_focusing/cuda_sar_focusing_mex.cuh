/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cudBackProjection_mex.cuh
 * Author: arwillis
 *
 * Created on June 22, 2021, 12:12 PM
 */

#ifndef CUDABACKPROJECTION_MEX_CUH
#define CUDABACKPROJECTION_MEX_CUH

#include <mex.h>    // Matlab library includes
#include <matrix.h> // Matlab mxComplexSingle struct

#include "../../cpuBackProjection/cpuBackProjection_mex.hpp"

template<typename __nTp>
int cuda_SARFocusingAlgorithm(int nrhs, const mxArray* prhs[], Complex<__nTp> *output_image) {

    SAR_Aperture<__nTp> sar_aperture_data;
    SAR_ImageFormationParameters<__nTp> sar_image_params;
    import_MATLABMexArguments(nrhs, prhs, sar_aperture_data, sar_image_params);

    std::cout << sar_aperture_data << std::endl;

    sar_aperture_data.polarity_channel = 1;
    if (sar_aperture_data.sampleData.shape.size() > 2) {
        sar_aperture_data.format_GOTCHA = false;
    }
    // the dimensional index of the polarity index in the 
    // multi-dimensional array (for Sandia SPH SAR data)
    if (!sar_aperture_data.format_GOTCHA) {
        sar_aperture_data.polarity_dimension = 2;
    }

    initialize_SAR_Aperture_Data(sar_aperture_data);

    std::cout << sar_aperture_data << std::endl;

    CArray<__nTp> output_image_arr(sar_image_params.N_y_pix * sar_image_params.N_x_pix);

    std::cout << sar_image_params << std::endl;
    sar_image_params.update(sar_aperture_data);
    //SAR_ImageFormationParameters<float> sar_image_params = SAR_ImageFormationParameters<float>::create<float>(sar_aperture_data);
    std::cout << sar_image_params << std::endl;

    cuda_focus_SAR_image(sar_aperture_data, sar_image_params, output_image_arr);

    for (int i = 0; i < output_image_arr.size(); i++) {
        //std::cout << "I(" << i << ") = " << output_image_arr[i] << std::endl;
        output_image[i]._M_real = output_image_arr[i].real();
        output_image[i]._M_imag = output_image_arr[i].imag();
    }
    
    return EXIT_SUCCESS;
}


#endif /* CUDABACKPROJECTION_MEX_CUH */

