#include <iomanip>
#include <numeric>
#include <sstream>

#include <cxxopts.hpp>
#include <matio.h>

#include "cpuBackProjection.hpp"

// For details see: https://github.com/jarro2783/cxxopts

void cxxopts_integration(cxxopts::Options& options) {

    options.add_options()
            ("i,input", "Input file", cxxopts::value<std::string>())
            ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
            ("o,output", "Output file", cxxopts::value<std::string>())
            ("h,help", "Print usage")
            ;
}

std::string HARDCODED_SARDATA_PATH = "/home/arwillis/sar/";

std::string Sandia_RioGrande_fileprefix = HARDCODED_SARDATA_PATH + "Sandia/Rio_Grande_UUR_SAND2021-1834_O/SPH/PHX1T03_PS0008_PT0000";
// index here is 2 digit file index [00,...,10]
std::string Sandia_RioGrande_filepostfix = "";
std::string Sandia_Farms_fileprefix = HARDCODED_SARDATA_PATH + "Sandia/Farms_UUR_SAND2021-1835_O/SPH/0506P19_PS0020_PT0000";
// index here is 2 digit file index [00,...,09]
std::string Sandia_Farms_filepostfix = "_N03_M1";

// azimuth=[1,...,360] for all GOTCHA polarities=(HH,VV,HV,VH) and pass=[pass1,...,pass7] 
std::string GOTCHA_fileprefix = HARDCODED_SARDATA_PATH + "GOTCHA/Gotcha-CP-All/DATA/pass1/HH/data_3dsar_pass1_az";
// index here is 3 digit azimuth [001,...,360]
std::string GOTCHA_filepostfix = "_HH";

void initialize_Sandia_SPHRead(std::unordered_map<std::string, matvar_t*> &matlab_readvar_map) {
    matlab_readvar_map["sph_MATData.total_pulses"] = NULL;
    matlab_readvar_map["sph_MATData.preamble.ADF"] = NULL;
    //matlab_readvar_map["sph_MATData.Const"] = NULL;
    matlab_readvar_map["sph_MATData.Data.ChirpRate"] = NULL;
    matlab_readvar_map["sph_MATData.Data.ChirpRateDelta"] = NULL;
    matlab_readvar_map["sph_MATData.Data.SampleData"] = NULL;
    matlab_readvar_map["sph_MATData.Data.StartF"] = NULL;
    matlab_readvar_map["sph_MATData.Data.radarCoordinateFrame.x"] = NULL;
    matlab_readvar_map["sph_MATData.Data.radarCoordinateFrame.y"] = NULL;
    matlab_readvar_map["sph_MATData.Data.radarCoordinateFrame.z"] = NULL;
    matlab_readvar_map["sph_MATData.Data.VelEast"] = NULL;
    matlab_readvar_map["sph_MATData.Data.VelNorth"] = NULL;
    matlab_readvar_map["sph_MATData.Data.VelDown"] = NULL;
    matlab_readvar_map["sph_MATData.Data.RxPos.xat"] = NULL;
    matlab_readvar_map["sph_MATData.Data.RxPos.yon"] = NULL;
    matlab_readvar_map["sph_MATData.Data.RxPos.zae"] = NULL;
    // data.phdata = sphObj.Data.SampleData(:,:,channelIndex);
    // % 1 = HH, 2 = HV, 3 = VH, 4 = VV
    // data.vfreq = zeros(numSamples, numPulses);
    // data.freq = data.vfreq(:,1);
    // data.AntX = sphObj.Data.radarCoordinateFrame.x(pulseIndices);
    // data.AntY = sphObj.Data.radarCoordinateFrame.y(pulseIndices);
    // data.AntZ = sphObj.Data.radarCoordinateFrame.z(pulseIndices);
    // chirpRateDelta = sphObj.Data.ChirpRateDelta(:, channelIndex);
    // startF = sphObj.Data.StartF(:, channelIndex);
    // startF = sphObj.Data.StartF(pulseIndex);
    // freq_per_sec = sphObj.Data.ChirpRate(pulseIndex);
    // freq_pre_sec_sq = sphObj.Data.ChirpRateDelta(pulseIndex);
    // wgs84 = wgs84Ellipsoid('kilometer');
    // antennaPos_ecef = zeros(numPulses,3);    
    // velocity(pulseIndex,:) = [sphObj.Data.VelEast(pulseIndex), ...
    //        sphObj.Data.VelNorth(pulseIndex), ...
    //        sphObj.Data.VelDown(pulseIndex)];
    //        
    //    % antenna phase center offset from the radar
    // antennaPos_geodetic = [sphObj.Data.RxPos.xat(pulseIndex), ...
    //        sphObj.Data.RxPos.yon(pulseIndex), ...
    //        sphObj.Data.RxPos.zae(pulseIndex)];        
    //    % Earth-Centered Earth-Fixed (ECEF)
    //    [antX, antY, antZ] = geodetic2ecef(wgs84, antennaPos_geodetic(1), ...
    //        antennaPos_geodetic(2), antennaPos_geodetic(3));
    //    antennaPos_ecef(pulseIndex,:) = [antX, antY, antZ];
    //    freq_per_sample(pulseIndex) = freq_per_sec/sphObj.preamble.ADF; % freq_per_sample    
    // freq_per_sec = sphObj.Data.ChirpRate(pulseIndex);
    // freq_pre_sec_sq = sphObj.Data.ChirpRateDelta(pulseIndex);
    // analogToDigitalConverterFrequency = sphObj.preamble.ADF; % Hertz
    // Ts0 = 1.0/analogToDigitalConverterFrequency;
    // chirpRates_rad = sphObj.Data.ChirpRate(1, pulseIndices, channelIndex)*pi/180;
    // nominalChirpRate = mean(chirpRates_rad);
    // centerFreq_rad = sphObj.preamble.DesCntrFreq*pi/180;
    // nominalChirpRate_rad = nominalChirpRate*pi/180;

}

void initialize_GOTCHA_MATRead(std::unordered_map<std::string, matvar_t*> &matlab_readvar_map) {
    matlab_readvar_map["data.fp"] = NULL;
    matlab_readvar_map["data.freq"] = NULL;
    matlab_readvar_map["data.x"] = NULL;
    matlab_readvar_map["data.y"] = NULL;
    matlab_readvar_map["data.z"] = NULL;
    matlab_readvar_map["data.r0"] = NULL;
    matlab_readvar_map["data.th"] = NULL;
    matlab_readvar_map["data.phi"] = NULL;
    matlab_readvar_map["data.af.r_correct"] = NULL;
    matlab_readvar_map["data.af.ph_correct"] = NULL;
}

#define VARPATH(stringvec) std::accumulate(stringvec.begin(), stringvec.end(), std::string(""))

bool read_MAT_Struct(mat_t* matfp, matvar_t * struct_matVar, std::unordered_map<std::string, matvar_t*> &matlab_readvar_map, std::vector<std::string>& context) {
    context.push_back(".");
    unsigned nFields = Mat_VarGetNumberOfFields(struct_matVar);
    for (int fieldIdx = 0; fieldIdx < nFields; fieldIdx++) {
        matvar_t* struct_fieldVar = Mat_VarGetStructFieldByIndex(struct_matVar, fieldIdx, 0);
        if (struct_fieldVar != NULL) {
            std::string varName(struct_fieldVar->name);
            if (struct_fieldVar->data_type == matio_types::MAT_T_STRUCT) {
                context.push_back(varName);
                read_MAT_Struct(matfp, struct_fieldVar, matlab_readvar_map, context);
                context.pop_back();
            } else {
                std::string current_fieldname = VARPATH(context) + varName;
                std::cout << current_fieldname << std::endl;
                for (std::unordered_map<std::string, matvar_t*>::iterator it = matlab_readvar_map.begin();
                        it != matlab_readvar_map.end(); ++it) {
                    std::string searched_fieldname = it->first;
                    //matvar_t *searched_matvar = it->second;
                    if (searched_fieldname == current_fieldname) {
                        std::cout << "Reading " << current_fieldname << " from file..." << std::endl;
                        int read_err = Mat_VarReadDataAll(matfp, struct_fieldVar);
                        if (read_err) {
                            std::cout << "Error reading data for variable " << current_fieldname << ". Exiting read process." << std::endl;
                            return false;
                        } else {
                            matlab_readvar_map[searched_fieldname] = struct_fieldVar;
                            Mat_VarPrint(struct_fieldVar, 1);
                        }
                    }
                }
            }
        }
    }
    context.pop_back();
    return true;
}

bool read_MAT_Variables(std::string inputfile, std::unordered_map<std::string, matvar_t*> &matlab_readvar_map) {
    mat_t *matfp = Mat_Open(inputfile.c_str(), MAT_ACC_RDONLY);
    std::vector<std::string> context;
    if (matfp) {

        matvar_t* matvar = Mat_VarReadInfo(matfp, "data.fp");
        if (NULL == matvar) {
            fprintf(stderr, "Variable ’data.fp’ not found, or error "
                    "reading MAT file\n");
        } else {
            if (!matvar->isComplex)
                fprintf(stderr, "Variable ’data.fp’ is not complex!\n");
            if (matvar->rank != 2 ||
                    (matvar->dims[0] > 1 && matvar->dims[1] > 1))
                fprintf(stderr, "Variable ’data.fp’ is not a vector!\n");
            Mat_VarFree(matvar);
        }

        matvar_t* root_matVar;
        while ((root_matVar = Mat_VarReadNext(matfp)) != NULL) {
            std::string varName(root_matVar->name);
            if (root_matVar->data_type == matio_types::MAT_T_STRUCT) {
                context.push_back(varName);
                read_MAT_Struct(matfp, root_matVar, matlab_readvar_map, context);
            } else if (root_matVar->data_type == matio_types::MAT_T_CELL) {
                std::cout << VARPATH(context) + varName << "is data of type MAT_T_CELL and cannot be read." << std::endl;
            } else {
                std::cout << VARPATH(context) + varName << " reading data..." << std::endl;
                int read_err = Mat_VarReadDataAll(matfp, root_matVar);
                if (read_err) {
                    //fprintf(stderr,"Error reading data for 'ing{%lu}.%s'\n",ing_index,ing_fieldname);
                    //err = EXIT_FAILURE;
                } else {
                    Mat_VarPrint(root_matVar, 1);
                }
            }
            Mat_VarFree(root_matVar);
            root_matVar = NULL;
        }
        Mat_Close(matfp);
    } else {
        std::cout << "Could not open MATLAB file " << inputfile << "." << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    Complex test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    Complex out[8];
    CArray data(test, 8);
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
        initialize_Sandia_SPHRead(matlab_readvar_map);

        // GOTCHA SAR DATA FILE LOADING
        //        int azimuth = 1; // 1-360 for all GOTCHA polarities=(HH,VV,HV,VH) and pass=[pass1,...,pass7] 
        //        std::string fileprefix = GOTCHA_fileprefix;
        //        std::string filepostfix = GOTCHA_filepostfix;
        //        ss << std::setfill('0') << std::setw(3) << azimuth;
        //        initialize_GOTCHA_MATRead(matlab_readvar_map);

        inputfile = fileprefix + ss.str() + filepostfix + ".mat";
    }

    std::cout << "Successfully opened MATLAB file " << inputfile << "." << std::endl;

    if (!read_MAT_Variables(inputfile, matlab_readvar_map)) {
        std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
        return EXIT_FAILURE;
    }
    // forward fft
    //fft(data);
    fftw(data);
    int N = 8;
    //    std::cout << "st " << sizeof (test) << std::endl;

    // http://www.fftw.org/fftw3_doc/Complex-numbers.html
    // Structure must be only two numbers in the order real, imag
    // to be binary compatible with the C99 complex type

    //    std::cout << "fft" << std::endl;
    //    for (int i = 0; i < 8; ++i) {
    //        std::cout << data[i] << std::endl;
    //    }

    // inverse fft
    //ifft(data);
    ifftw(data);

    //    std::cout << std::endl << "ifft" << std::endl;
    //    for (int i = 0; i < 8; ++i) {
    //        std::cout << data[i] << std::endl;
    //    }
    return EXIT_SUCCESS;
}

/* Main kernel.
 *
 * Tuning options:
 * - is it worth #defining radar parameters like start_frequency?
 *      ............  or imaging parameters like xmin/ymax?
 * - Make sure (4 pi / c) is computed at compile time!
 * - Use 24-bit integer multiplications!
 *
 * */

//void backprojection_loop(float2* full_image,
//        int Npulses, int Ny_pix, float delta_x_m_per_pix, float delta_y_m_per_pix,
//        int PROJ_LENGTH,
//        int X_OFFSET, int Y_OFFSET,
//        float C__4_DELTA_FREQ, float* PI_4_F0__CLIGHT,
//        float left, float bottom, float min_eff_idx, float4 * platform_info,
//        float * debug_effective_idx, float * debug_2, float * x_mat, float * y_mat,
//        float rmin, float rmax) {
//
//
//}

//void from_gpu_complex_to_bp_complex_split(float2 * data, bp_complex_split out, int size) {
//    int i;
//    for (i = 0; i < size; i++) {
//        out.real[i] = data[i].x;
//        out.imag[i] = data[i].y;
//    }
//}
//
//float2* format_complex_to_columns(bp_complex_split a, int width_orig, int height_orig) {
//    float2* out = (float2*) malloc(width_orig * height_orig * sizeof (float2));
//    int i, j;
//    for (i = 0; i < height_orig; i++) {
//        int origOffset = i * width_orig;
//        for (j = 0; j < width_orig; j++) {
//            int newOffset = j * height_orig;
//            out[newOffset + i].x = a.real[origOffset + j];
//            out[newOffset + i].y = a.imag[origOffset + j];
//        }
//    }
//    return out;
//}
//
//float2* format_complex(bp_complex_split a, int size) {
//    float2* out = (float2*) malloc(size * sizeof (float2));
//    int i;
//    for (i = 0; i < size; i++) {
//        out[i].x = a.real[i];
//        out[i].y = a.imag[i];
//    }
//    return out;
//}
//
//float4* format_x_y_z_r(float * x, float * y, float * z, float * r, int size) {
//    float4* out = (float4*) malloc(size * sizeof (float4));
//    int i;
//    for (i = 0; i < size; i++) {
//        out[i].x = x[i];
//        out[i].y = y[i];
//        out[i].z = z[i];
//        out[i].w = r[i];
//    }
//    return out;
//}

/* Credits: from BackProjectionKernal.c: "originally by reinke".
 * Given a float X, returns float2 Y = exp(j * X).
 *
 * __device__ code is always inlined. */

//float2 expjf(float in) {
//    float2 out;
//    float t, tb;
//#if USE_FAST_MATH
//    t = __tanf(in / 2.0f);
//#else
//    t = tan(in / 2.0f);
//#endif
//    tb = t * t + 1.0f;
//    out.x = (2.0f - tb) / tb; /* Real */
//    out.y = (2.0f * t) / tb; /* Imag */
//
//    return out;
//}
//
//float2 expjf_div_2(float in) {
//    float2 out;
//    float t, tb;
//    //t = __tanf(in - (float)((int)(in/(PI2)))*PI2 );
//    t = std::tan(in - PI * std::round(in / PI));
//    tb = t * t + 1.0f;
//    out.x = (2.0f - tb) / tb; /* Real */
//    out.y = (2.0f * t) / tb; /* Imag */
//    return out;
//}