#include <iomanip>
#include <sstream>

#include <cxxopts.hpp>

#include "cpuBackProjection.hpp"
#include "cpuBackProjection_main.hpp"

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

void initialize_GOTCHA_MATRead(std::unordered_map<std::string, matvar_t*>& matlab_readvar_map) {
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

void initialize_Sandia_SPHRead(std::unordered_map<std::string, matvar_t*> &matlab_readvar_map) {
    matlab_readvar_map["sph_MATData.total_pulses"] = NULL;
    matlab_readvar_map["sph_MATData.preamble.ADF"] = NULL;
    matlab_readvar_map["sph_MATData.Data.ChirpRate"] = NULL;
    matlab_readvar_map["sph_MATData.Data.ChirpRateDelta"] = NULL;
    matlab_readvar_map["sph_MATData.Data.SampleData"] = NULL;
    matlab_readvar_map["sph_MATData.Data.StartF"] = NULL;
    matlab_readvar_map["sph_MATData.Data.radarCoordinateFrame.x"] = NULL;
    matlab_readvar_map["sph_MATData.Data.radarCoordinateFrame.y"] = NULL;
    matlab_readvar_map["sph_MATData.Data.radarCoordinateFrame.z"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.VelEast"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.VelNorth"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.VelDown"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.RxPos.xat"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.RxPos.yon"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.RxPos.zae"] = NULL;
}

// For details see: https://github.com/jarro2783/cxxopts

void cxxopts_integration(cxxopts::Options& options) {

    options.add_options()
            ("i,input", "Input file", cxxopts::value<std::string>())
            //("f,format", "Data format {GOTCHA, Sandia, <auto>}", cxxopts::value<std::string>()->default_value("auto"))
            ("p,polarity", "Polarity {HH,HV,VH,VV,<any>}", cxxopts::value<std::string>()->default_value("any"))
            ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
            ("o,output", "Output file", cxxopts::value<std::string>())
            ("h,help", "Print usage")
            ;
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

        initialize_GOTCHA_MATRead(matlab_readvar_map);

        inputfile = fileprefix + ss.str() + filepostfix + ".mat";
    }

    std::cout << "Successfully opened MATLAB file " << inputfile << "." << std::endl;

    SAR_Aperture<float> aperture_data;
    if (read_MAT_Variables(inputfile, matlab_readvar_map, aperture_data) == EXIT_FAILURE) {
        std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << aperture_data << std::endl;

    // Sandia SAR data is multi-channel having up to 4 polarities
    // 1 = HH, 2 = HV, 3 = VH, 4 = VVbandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
    if (result.count("polarity")) {
        std::string polarity = result["polarity"].as<std::string>();
        if (polarity == "HH" && aperture_data.sampleData.shape.size() >= 1) {
            aperture_data.polarity_channel = 1;
        } else if (polarity == "HV" && aperture_data.sampleData.shape.size() >= 2) {
            aperture_data.polarity_channel = 2;
        } else if (polarity == "VH" && aperture_data.sampleData.shape.size() >= 3) {
            aperture_data.polarity_channel = 3;
        } else if (polarity == "VV" && aperture_data.sampleData.shape.size() >= 4) {
            aperture_data.polarity_channel = 4;
        } else {
            std::cout << "Request polarity channel " << polarity << " is not available." << std::endl;
            return EXIT_FAILURE;
        }
    }

    initializeSARFocusingVariables(aperture_data);

    std::cout << aperture_data << std::endl;
    //finalize_GOTCHA_MATRead(matlab_readvar_map, aperture_data);
    //finalize_GOTCHA_MATRead(aperture_data, matlab_readvar_map);
    //for (std::unordered_map<std::string, matvar_t*>::iterator it = matlab_readvar_map.begin();
    //        it != matlab_readvar_map.end(); ++it) {
    //std::string searched_fieldname = it->first;
    //matvar_t* fieldVar = it->second;

    //fieldVar->data_type == matio_types::MAT_T_DOUBLE
    //}
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
