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
//#include <matrix.h> // Matlab mxComplexSingle struct

#include "cpuBackProjection.hpp"

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
        mxComplexSingleClass<double>* complex_data = reinterpret_cast<mxComplexSingleClass<double> *> (mxGetComplexDoubles(matArg));
        for (int idx = 0; idx < totalsize; idx++) {
            sMat.data.push_back(__Tp(complex_data[idx].real, complex_data[idx].imag));
        }
    } else if (mxIsSingle(matArg)) {
        mxComplexSingleClass<float>* complex_data = reinterpret_cast<mxComplexSingleClass<float> *> (mxGetComplexSingles(matArg));
        for (int idx = 0; idx < totalsize; idx++) {
            sMat.data.push_back(__Tp(complex_data[idx].real, complex_data[idx].imag));
        }
    } else {
        std::cout << "import_MATLABArgumentComplex::Matrix representation must be double-precision or single-precision." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

template<typename __Tp1, typename __Tp2>
int importMATLABMexArguments(int nrhs, const mxArray* prhs[],
        SAR_Aperture<__Tp1>& sar_aperture_data,
        SAR_ImageFormationParameters<__Tp2>& sar_image_params) {

    sar_aperture_data.numAzimuthSamples = mxGetN(prhs[0]);
    sar_aperture_data.numRangeSamples = mxGetM(prhs[0]);
    import_MATLABArgumentComplex(prhs[0], sar_aperture_data.sampleData);

    import_MATLABArgumentReal(prhs[1], sar_aperture_data.freq);
    import_MATLABArgumentReal(prhs[2], sar_aperture_data.Ant_x);
    import_MATLABArgumentReal(prhs[3], sar_aperture_data.Ant_y);
    import_MATLABArgumentReal(prhs[4], sar_aperture_data.Ant_z);
    import_MATLABArgumentReal(prhs[5], sar_aperture_data.slant_range);
    
    sar_image_params.N_x_pix = (int) mxGetScalar(prhs[6]);
    sar_image_params.N_y_pix = (int) mxGetScalar(prhs[7]);
    sar_image_params.N_fft = (int) mxGetScalar(prhs[8]);
    sar_image_params.x0_m = (float) mxGetScalar(prhs[9]);
    sar_image_params.y0_m = (float) mxGetScalar(prhs[10]);
    sar_image_params.Wx_m = (float) mxGetScalar(prhs[11]);
    sar_image_params.Wy_m = (float) mxGetScalar(prhs[12]);

    return EXIT_SUCCESS;
}

#endif /* CPUBACKPROJECTION_MEX_HPP */

