// this declaration needs to be in any C++ compiled target for CPU
#define CUDAFUNCTION __host__ __device__

// Standard Library includes
#include <stdio.h>  /* printf */
#include <time.h>

// MATLAB includes
#include <mex.h>  
#include <gpu/mxGPUArray.h>

#include "cuda_sar_focusing.hpp"
#include "cuda_sar_focusing_mex.cuh"

void mexFunction(int nlhs, /* number of LHS (output) arguments */
        mxArray* plhs[], /* array of mxArray pointers to outputs */
        int nrhs, /* number of RHS (input) args */
        const mxArray* prhs[]) /* array of pointers to inputs*/ {

    /* Check that phase history inputs is complex*/
    if (!mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("mexFunction::Input 0 is not complex-valued",
                "Input 0 must be complex-valued.\n");
        plhs[0] = mxCreateNumericMatrix(0, 0,
                mxDOUBLE_CLASS, mxCOMPLEX);
        return;
    }

    int N_x_pix = (int) mxGetScalar(prhs[ARG_N_X_PIX]);
    int N_y_pix = (int) mxGetScalar(prhs[ARG_N_Y_PIX]);

    // Initialize the MathWorks GPU API
    mxInitGPU();

    if (mxIsDouble(prhs[0])) {
                std::cout << "Running in double precision mode." << std::endl;
        plhs[0] = mxCreateNumericMatrix(N_y_pix, N_x_pix,
                mxDOUBLE_CLASS, mxCOMPLEX);
        mxComplexDouble* output_image = mxGetComplexDoubles(plhs[0]);
        // MATLAB mxComplexDouble is only a C struct with *zero* arithmetic support.
        // I must unfortunately cast the memory to a class to operate on the data using
        // Object-Oriented Programming abstraction. *NOT A DESIRABLE CAST*
        Complex<double>* output_image_cast = reinterpret_cast<Complex<double>*> (output_image);

        cuda_SARFocusingAlgorithm<double>(nrhs, prhs, output_image_cast);
    } else {
        std::cout << "Running in single precision mode." << std::endl;
        plhs[0] = mxCreateNumericMatrix(N_y_pix, N_x_pix,
                mxSINGLE_CLASS, mxCOMPLEX);
        mxComplexSingle* output_image = mxGetComplexSingles(plhs[0]);
        // MATLAB mxComplexSingle is only a C struct with *zero* arithmetic support.
        // I must unfortunately cast the memory to a class to operate on the data using
        // Object-Oriented Programming abstraction. *NOT A DESIRABLE CAST*
        Complex<float>* output_image_cast = reinterpret_cast<Complex<float>*> (output_image);

        cuda_SARFocusingAlgorithm<float>(nrhs, prhs, output_image_cast);
    }
}
