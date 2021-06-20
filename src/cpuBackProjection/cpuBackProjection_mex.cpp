
#ifndef NO_MATLAB

#include "cpuBackProjection_mex.hpp"

void mexFunction(int nlhs, /* number of LHS (output) arguments */
        mxArray* plhs[], /* array of mxArray pointers to outputs */
        int nrhs, /* number of RHS (input) args */
        const mxArray* prhs[]) /* array of pointers to inputs*/ {

    mxComplexSingle* range_profiles;


    /* Check that phase history inputs is complex*/
    if (!mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("mexFunction::Input 0 is not complex-valued",
                "Input 0 must be complex-valued.\n");
    }
    
    /* setup Matlab output */
    if (mxIsDouble(prhs[0])) {
        std::cout << "Running in double precision mode." << std::endl;
        mxComplexDouble* output_image;
        SAR_Aperture<double> sar_aperture_data;
        SAR_ImageFormationParameters<double> sar_image_params;
        importMATLABMexArguments(nrhs, prhs, sar_aperture_data, sar_image_params);

        std::cout << sar_aperture_data << std::endl;

        sar_aperture_data.polarity_channel = 1;
        // the dimensional index of the polarity index in the 
        // multi-dimensional array (for Sandia SPH SAR data)
        if (!sar_aperture_data.format_GOTCHA) {
            sar_aperture_data.polarity_dimension = 2;
        }

        initialize_SAR_Aperture_Data(sar_aperture_data);

        std::cout << sar_aperture_data << std::endl;

        plhs[0] = mxCreateNumericMatrix(sar_image_params.N_y_pix, sar_image_params.N_x_pix,
                mxDOUBLE_CLASS, mxCOMPLEX);
        output_image = mxGetComplexDoubles(plhs[0]);
        mxComplexSingleClass<double>* range_profiles_cast = reinterpret_cast<mxComplexSingleClass<double>*> (range_profiles);
        mxComplexSingleClass<double>* output_image_cast = reinterpret_cast<mxComplexSingleClass<double>*> (output_image);
        std::valarray<mxComplexSingleClass<double>> output_image_arr(sar_image_params.N_y_pix * sar_image_params.N_x_pix);

        std::cout << sar_image_params << std::endl;
        sar_image_params.update(sar_aperture_data);
        //SAR_ImageFormationParameters<double> sar_image_params = SAR_ImageFormationParameters<double>::create<double>(sar_aperture_data);
        std::cout << sar_image_params << std::endl;

//        focus_SAR_image(sar_aperture_data, sar_image_params, output_image_arr);

        for (int i = 0; i < output_image_arr.size(); i++) {
            //std::cout << "I(" << i << ") = " << output_image_arr[i] << std::endl;
            output_image[i].real = output_image_arr[i].real;
            output_image[i].imag = output_image_arr[i].imag;
        }
    } else {
        std::cout << "Running in single precision mode." << std::endl;
        mxComplexSingle* output_image;
        SAR_Aperture<float> sar_aperture_data;
        SAR_ImageFormationParameters<float> sar_image_params;
        importMATLABMexArguments(nrhs, prhs, sar_aperture_data, sar_image_params);

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

        plhs[0] = mxCreateNumericMatrix(sar_image_params.N_y_pix, sar_image_params.N_x_pix,
                mxSINGLE_CLASS, mxCOMPLEX);
        output_image = mxGetComplexSingles(plhs[0]);
        mxComplexSingleClass<float>* output_image_cast = reinterpret_cast<mxComplexSingleClass<float>*> (output_image);
        std::valarray<mxComplexSingleClass<float>> output_image_arr(sar_image_params.N_y_pix * sar_image_params.N_x_pix);

        std::cout << sar_image_params << std::endl;
        sar_image_params.update(sar_aperture_data);
        //SAR_ImageFormationParameters<float> sar_image_params = SAR_ImageFormationParameters<float>::create<float>(sar_aperture_data);
        std::cout << sar_image_params << std::endl;

        focus_SAR_image(sar_aperture_data, sar_image_params, output_image_arr);

        for (int i = 0; i < output_image_arr.size(); i++) {
            //std::cout << "I(" << i << ") = " << output_image_arr[i] << std::endl;
            output_image[i].real = output_image_arr[i].real;
            output_image[i].imag = output_image_arr[i].imag;
        }
    }
    return;
}
#endif
