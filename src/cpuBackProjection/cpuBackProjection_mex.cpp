
#include "cpuBackProjection.hpp"

#ifndef NO_MATLAB

void mexFunction(int nlhs, /* number of LHS (output) arguments */
        mxArray* plhs[], /* array of mxArray pointers to outputs */
        int nrhs, /* number of RHS (input) args */
        const mxArray* prhs[]) /* array of pointers to inputs*/ {
    mxComplexSingle* range_profiles;
    float* minF;
    float* aimpoint_ranges;
    float* xobs;
    float* yobs;
    float* zobs;
    int Nx_pix, Ny_pix;
    float* deltaF;
    float x0, y0, Wx, Wy; // image (x,y) center and (width,height) w.r.t. target phase center

    float min_eff_idx; //, Nrangebins;

    int Npulses, Nrangebins, Nfft;

    mxComplexSingle* output_image;

    /* Section 2. 
     * Parse Matlab's inputs */
    /* Check that both inputs are complex*/
    if (!mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:convec:inputsNotComplex",
                "Input 0 must be complex.\n");
    }

    /* Range profile dimensions */
    Npulses = mxGetN(prhs[0]);
    Nrangebins = mxGetM(prhs[0]);

    range_profiles = mxGetComplexSingles(prhs[0]);

    //range_profiles.real = (float*) mxGetPr(prhs[0]);
    //range_profiles.imag = (float*) mxGetPi(prhs[0]);

    minF = (float*) mxGetPr(prhs[1]);
    deltaF = (float*) mxGetPr(prhs[2]);

    aimpoint_ranges = (float*) mxGetPr(prhs[3]);
    xobs = (float*) mxGetPr(prhs[4]);
    yobs = (float*) mxGetPr(prhs[5]);
    zobs = (float*) mxGetPr(prhs[6]);

    Nx_pix = (int) mxGetScalar(prhs[7]);
    Ny_pix = (int) mxGetScalar(prhs[8]);

    Nfft = (int) mxGetScalar(prhs[9]);
    x0 = (float) mxGetScalar(prhs[10]);
    y0 = (float) mxGetScalar(prhs[11]);
    Wx = (float) mxGetScalar(prhs[12]);
    Wy = (float) mxGetScalar(prhs[13]);

    /* Setup some intermediate values */

    if (nrhs == 16) {
        min_eff_idx = (float) mxGetScalar(prhs[14]);
        Nrangebins = (float) mxGetScalar(prhs[15]);
    } else {
        min_eff_idx = 0;
        //Nrangebins = Nrangebins;
    }

    /* setup Matlab output */
    plhs[0] = mxCreateNumericMatrix(Ny_pix, Nx_pix, mxSINGLE_CLASS, mxCOMPLEX);
    output_image = mxGetComplexSingles(plhs[0]);
    //output_image.real = (float*) mxGetPr(plhs[0]);
    //output_image.imag = (float*) mxGetPi(plhs[0]);
    mxComplexSingleClass* range_profiles_cast = static_cast<mxComplexSingleClass*> (range_profiles);
    mxComplexSingleClass* output_image_cast = static_cast<mxComplexSingleClass*> (output_image);

    CArray range_profiles_arr(range_profiles_cast, Npulses * Nrangebins);
    //CArray range_profiles_arr(Npulses * Nrangebins);
    CArray output_image_arr(output_image_cast, Ny_pix * Nx_pix);
    //    for (int i = 0; i < Npulses * Nrangebins; i++) {
    //        //std::cout << "I(" << i << ") = " << output_image_arr[i] << std::endl;
    //        range_profiles_arr[i] = range_profiles[i];
    //    }

    run_bp(range_profiles_arr, xobs, yobs, zobs,
            aimpoint_ranges,
            Npulses, Nrangebins, Nx_pix, Ny_pix, Nfft,
            output_image_arr,
            minF, deltaF,
            x0, y0, Wx, Wy, min_eff_idx, Nrangebins);
    for (int i = 0; i < output_image_arr.size(); i++) {
        //std::cout << "I(" << i << ") = " << output_image_arr[i] << std::endl;
        output_image[i] = output_image_arr[i];
    }

    return;
}
#endif
