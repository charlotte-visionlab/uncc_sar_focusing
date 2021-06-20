/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cpuBackProjection_mex.hpp
 * Author: arwillis
 *
 * Created on June 19, 2021, 7:56 PM
 */

#ifndef CPUBACKPROJECTION_MEX_HPP
#define CPUBACKPROJECTION_MEX_HPP

#include <mex.h>    // Matlab library includes
#include <matrix.h> // Matlab mxComplexSingle struct

#include "uncc_sar_focusing.hpp"

#define ARG_FREQ 1
#define ARG_ANT_X 2
#define ARG_ANT_Y 3
#define ARG_ANT_Z 4
#define ARG_SLANT_RANGE 5
#define ARG_N_X_PIX 6
#define ARG_N_Y_PIX 7
#define ARG_N_FFT 8
#define ARG_X0_M 9
#define ARG_Y0_M 10
#define ARG_WX_M 11
#define ARG_WY_M 12

template<typename __Tp>
int import_MATLABArgumentReal(const mxArray* matArg, simpleMatrix<__Tp>& sMat) {
    const mwSize ndims = mxGetNumberOfDimensions(matArg);
    const mwSize *dims = mxGetDimensions(matArg);
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        sMat.shape.push_back(dims[dimIdx]);
        totalsize = totalsize * dims[dimIdx];
    }
    if (mxIsComplex(matArg)) {
        std::cout << "import_MATLABArgumentReal::Matrix is complex-valued and not real-valued!" << std::endl;
        return EXIT_FAILURE;
    }
    if (mxIsDouble(matArg)) {
        double* real_data = reinterpret_cast<double *> (mxGetDoubles(matArg));
        for (int idx = 0; idx < totalsize; idx++) {
            sMat.data.push_back(real_data[idx]);
        }
    } else if (mxIsSingle(matArg)) {
        float* real_data = reinterpret_cast<float *> (mxGetSingles(matArg));
        for (int idx = 0; idx < totalsize; idx++) {
            sMat.data.push_back(real_data[idx]);
        }
    } else {
        std::cout << "import_MATLABArgumentReal::Matrix representation must be double-precision or single-precision." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

template<typename __Tp>
int import_MATLABArgumentComplex(const mxArray* matArg, simpleMatrix<__Tp>& sMat) {
    const mwSize ndims = mxGetNumberOfDimensions(matArg);
    const mwSize *dims = mxGetDimensions(matArg);
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        sMat.shape.push_back(dims[dimIdx]);
        totalsize = totalsize * dims[dimIdx];
    }
    if (!mxIsComplex(matArg)) {
        std::cout << "import_MATLABArgumentComplex::Matrix is real-valued and not complex-valued!" << std::endl;
        return EXIT_FAILURE;
    }
    if (mxIsDouble(matArg)) {
        Complex<double>* complex_data = reinterpret_cast<Complex<double> *> (mxGetComplexDoubles(matArg));
        for (int idx = 0; idx < totalsize; idx++) {
            sMat.data.push_back(__Tp(complex_data[idx].real(), complex_data[idx].imag()));
        }
    } else if (mxIsSingle(matArg)) {
        Complex<float>* complex_data = reinterpret_cast<Complex<float> *> (mxGetComplexSingles(matArg));
        for (int idx = 0; idx < totalsize; idx++) {
            sMat.data.push_back(__Tp(complex_data[idx].real(), complex_data[idx].imag()));
        }
    } else {
        std::cout << "import_MATLABArgumentComplex::Matrix representation must be double-precision or single-precision." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

template<typename __Tp1, typename __Tp2>
int import_MATLABMexArguments(int nrhs, const mxArray* prhs[],
        SAR_Aperture<__Tp1>& sar_aperture_data,
        SAR_ImageFormationParameters<__Tp2>& sar_image_params) {

    sar_aperture_data.numAzimuthSamples = mxGetN(prhs[0]);
    sar_aperture_data.numRangeSamples = mxGetM(prhs[0]);
    import_MATLABArgumentComplex(prhs[0], sar_aperture_data.sampleData);

    import_MATLABArgumentReal(prhs[ARG_FREQ], sar_aperture_data.freq);
    import_MATLABArgumentReal(prhs[ARG_ANT_X], sar_aperture_data.Ant_x);
    import_MATLABArgumentReal(prhs[ARG_ANT_Y], sar_aperture_data.Ant_y);
    import_MATLABArgumentReal(prhs[ARG_ANT_Z], sar_aperture_data.Ant_z);
    import_MATLABArgumentReal(prhs[ARG_SLANT_RANGE], sar_aperture_data.slant_range);

    sar_image_params.N_x_pix = (int) mxGetScalar(prhs[ARG_N_X_PIX]);
    sar_image_params.N_y_pix = (int) mxGetScalar(prhs[ARG_N_Y_PIX]);
    sar_image_params.N_fft = (int) mxGetScalar(prhs[ARG_N_FFT]);
    sar_image_params.x0_m = (float) mxGetScalar(prhs[ARG_X0_M]);
    sar_image_params.y0_m = (float) mxGetScalar(prhs[ARG_Y0_M]);
    sar_image_params.Wx_m = (float) mxGetScalar(prhs[ARG_WX_M]);
    sar_image_params.Wy_m = (float) mxGetScalar(prhs[ARG_WY_M]);

    return EXIT_SUCCESS;
}

template<typename __nTp>
int runSARFocusingAlgorithm(int nrhs, const mxArray* prhs[], Complex<__nTp> *output_image) {

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

    focus_SAR_image(sar_aperture_data, sar_image_params, output_image_arr);

    for (int i = 0; i < output_image_arr.size(); i++) {
        //std::cout << "I(" << i << ") = " << output_image_arr[i] << std::endl;
        output_image[i]._M_real = output_image_arr[i].real();
        output_image[i]._M_imag = output_image_arr[i].imag();
    }
    
    return EXIT_SUCCESS;
}

#endif /* CPUBACKPROJECTION_MEX_HPP */

