/* 
 * File:   cpuBackProjection_main.hpp
 * Author: arwillis
 *
 * Created on June 17, 2021, 10:00 PM
 */

#ifndef CPUBACKPROJECTION_MAIN_HPP
#define CPUBACKPROJECTION_MAIN_HPP

#include <complex>
#include <numeric>

#include <matio.h>

#include "cpuBackProjection.hpp"

template<typename _numTp>
int import_MATVector(matvar_t* matVar, simpleMatrix<_numTp>& sMat) {
    int ndims = matVar->rank;
    int sizes[ndims];
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        sizes[ndims] = matVar->dims[dimIdx];
        sMat.shape.push_back(matVar->dims[dimIdx]);
        totalsize = totalsize * sizes[ndims];
    }
    char *dp = (char *) matVar->data;
    switch (matVar->class_type) {
        case matio_classes::MAT_C_SINGLE:
            sMat.data.insert(sMat.data.end(), (float *) dp, (float *) (dp + matVar->dims[0] * matVar->dims[1] * matVar->data_size));
            break;
        case matio_classes::MAT_C_DOUBLE:
            sMat.data.insert(sMat.data.end(), (double *) dp, (double *) (dp + matVar->dims[0] * matVar->dims[1] * matVar->data_size));
            break;
        default:
            std::cout << "import_MATVector::Type of data not recognized!" << std::endl;
    }
}

template<typename _realTp>
int import_MATMatrixReal(matvar_t* matVar, simpleMatrix<_realTp>& sMat) {
    int ndims = matVar->rank;
    int sizes[ndims];
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        sizes[ndims] = matVar->dims[dimIdx];
        sMat.shape.push_back(matVar->dims[dimIdx]);
        totalsize = totalsize * sizes[ndims];
    }
    size_t stride = matVar->data_size;

    if (matVar->isComplex) {
        std::cout << "import_MATMatrixReal::Matrix is complex-valued!" << std::endl;
        return EXIT_FAILURE;
    }
    char *dp = (char *) matVar->data;
    switch (matVar->class_type) {
        case matio_classes::MAT_C_SINGLE:
            sMat.data.insert(sMat.data.end(), (float *) dp, (float *) (dp + totalsize * matVar->data_size));
            break;
        case matio_classes::MAT_C_DOUBLE:
            sMat.data.insert(sMat.data.end(), (double *) dp, (double *) (dp + totalsize * matVar->data_size));
            break;
        default:
            std::cout << "import_MATVector::Type of data not recognized!" << std::endl;
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

template<typename _complexTp>
int import_MATMatrixComplex(matvar_t* matVar, simpleMatrix<_complexTp>& sMat) {
    int ndims = matVar->rank;
    int sizes[ndims];
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        sizes[ndims] = matVar->dims[dimIdx];
        sMat.shape.push_back(matVar->dims[dimIdx]);
        totalsize = totalsize * sizes[ndims];
    }
    size_t stride = matVar->data_size;
    if (!matVar->isComplex) {
        std::cout << "import_MATMatrixComplex::Matrix is not complex-valued!" << std::endl;
        return EXIT_FAILURE;
    }
    mat_complex_split_t* complex_data = (mat_complex_split_t *) matVar->data;
    char *rp = (char *) complex_data->Re;
    char *ip = (char *) complex_data->Im;
    switch (matVar->class_type) {
        case matio_classes::MAT_C_SINGLE:
            for (int idx = 0; idx < totalsize; idx++) {
                sMat.data.push_back(_complexTp(*(float*) (rp + idx * stride),
                        *(float*) (ip + idx * stride)));
            }
            break;
        case matio_classes::MAT_C_DOUBLE:
            for (int idx = 0; idx < totalsize; idx++) {
                sMat.data.push_back(_complexTp(*(double*) (rp + idx * stride),
                        *(double*) (ip + idx * stride)));
            }
            break;
        default:
            std::cout << "import_GOTCHA_MATData::Type of phase data not recognized!" << std::endl;
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

template<typename _numTp>
int import_Sandia_MATData(matvar_t* matVar, std::string fieldname, SAR_Aperture<_numTp>& aperture) {
    if (fieldname == "sph_MATData.Data.SampleData") {
        if (!matVar->isComplex) {
            std::cout << "import_GOTCHA_MATData::Phase data is not complex-valued!" << std::endl;
            return EXIT_FAILURE;
        }
        import_MATMatrixComplex(matVar, aperture.sampleData);
    } else if (fieldname == "sph_MATData.Data.StartF") {
        import_MATMatrixReal(matVar, aperture.startF);
    } else if (fieldname == "sph_MATData.Data.radarCoordinateFrame.x") {
        import_MATMatrixReal(matVar, aperture.Ant_x);
    } else if (fieldname == "sph_MATData.Data.radarCoordinateFrame.y") {
        import_MATMatrixReal(matVar, aperture.Ant_y);
    } else if (fieldname == "sph_MATData.Data.radarCoordinateFrame.z") {
        import_MATMatrixReal(matVar, aperture.Ant_z);
    } else if (fieldname == "sph_MATData.Data.ChirpRate") {
        import_MATMatrixReal(matVar, aperture.chirpRate);
    } else if (fieldname == "sph_MATData.Data.ChirpRateDelta") {
        import_MATMatrixReal(matVar, aperture.chirpRateDelta);
    } else if (fieldname == "sph_MATData.preamble.ADF") {
        import_MATMatrixReal(matVar, aperture.ADF);
    } else {
        std::cout << "import_Sandia_MATData::Fieldname " << fieldname << " not recognized.";
        return EXIT_FAILURE;
    }
    //    matlab_readvar_map["sph_MATData.Data.VelEast"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.VelNorth"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.VelDown"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.RxPos.xat"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.RxPos.yon"] = NULL;
    //    matlab_readvar_map["sph_MATData.Data.RxPos.zae"] = NULL;
    return EXIT_SUCCESS;
}

template<typename _numTp>
int import_GOTCHA_MATData(matvar_t* matVar, std::string fieldname, SAR_Aperture<_numTp>& aperture) {
    if (fieldname == "data.fp") {
        if (!matVar->isComplex) {
            std::cout << "import_GOTCHA_MATData::Phase data is not complex-valued!" << std::endl;
            return EXIT_FAILURE;
        }
        import_MATMatrixComplex(matVar, aperture.sampleData);
    } else if (fieldname == "data.freq") {
        import_MATMatrixReal(matVar, aperture.freq);
    } else if (fieldname == "data.x") {
        import_MATMatrixReal(matVar, aperture.Ant_x);
    } else if (fieldname == "data.y") {
        import_MATMatrixReal(matVar, aperture.Ant_y);
    } else if (fieldname == "data.z") {
        import_MATMatrixReal(matVar, aperture.Ant_z);
    } else if (fieldname == "data.r0") {
        import_MATMatrixReal(matVar, aperture.slant_range);
    } else if (fieldname == "data.th") {
        import_MATMatrixReal(matVar, aperture.theta);
    } else if (fieldname == "data.phi") {
        import_MATMatrixReal(matVar, aperture.phi);
    } else if (fieldname == "data.af.r_correct") {
        import_MATMatrixReal(matVar, aperture.af.r_correct);
    } else if (fieldname == "data.af.ph_correct") {
        import_MATMatrixReal(matVar, aperture.af.ph_correct);
    } else {
        std::cout << "import_GOTCHA_MATData::Fieldname " << fieldname << " not recognized.";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

#define CONCAT_STRINGVECTOR(stringvec) std::accumulate(stringvec.begin(), stringvec.end(), std::string(""))

template<typename _numTp>
int read_MAT_Struct(mat_t* matfp, matvar_t * struct_matVar,
        std::unordered_map<std::string, matvar_t*> &matlab_readvar_map,
        std::vector<std::string>& context,
        SAR_Aperture<_numTp>& aperture) {

    context.push_back(".");
    unsigned nFields = Mat_VarGetNumberOfFields(struct_matVar);
    for (int fieldIdx = 0; fieldIdx < nFields; fieldIdx++) {
        matvar_t* struct_fieldVar = Mat_VarGetStructFieldByIndex(struct_matVar, fieldIdx, 0);
        if (struct_fieldVar != NULL) {
            std::string varName(struct_fieldVar->name);
            if (struct_fieldVar->data_type == matio_types::MAT_T_STRUCT) {
                context.push_back(varName);
                read_MAT_Struct(matfp, struct_fieldVar, matlab_readvar_map, context, aperture);
                context.pop_back();
            } else {
                std::string current_fieldname = CONCAT_STRINGVECTOR(context) + varName;
                std::cout << current_fieldname << std::endl;
                for (std::unordered_map<std::string, matvar_t*>::iterator it = matlab_readvar_map.begin();
                        it != matlab_readvar_map.end(); ++it) {
                    std::string searched_fieldname = it->first;
                    if (searched_fieldname == current_fieldname) {
                        std::cout << "Reading " << current_fieldname << " from file..." << std::endl;
                        int read_err = Mat_VarReadDataAll(matfp, struct_fieldVar);
                        if (read_err) {
                            std::cout << "Error reading data for variable " << current_fieldname << ". Exiting read process." << std::endl;
                            return EXIT_FAILURE;
                        } else {
                            matlab_readvar_map[searched_fieldname] = struct_fieldVar;
                            //Mat_VarPrint(struct_fieldVar, 1);
                        }
                        if (current_fieldname.substr(0, 3) == "sph") {
                            import_Sandia_MATData(struct_fieldVar, current_fieldname, aperture);
                            aperture.format_GOTCHA = false;
                        } else {
                            import_GOTCHA_MATData(struct_fieldVar, current_fieldname, aperture);
                            aperture.format_GOTCHA = true;
                        }
                        //Mat_VarFree(struct_fieldVar);
                        //struct_fieldVar = NULL;
                    }
                }
            }
        }
    }
    context.pop_back();
    return EXIT_SUCCESS;
}

template<typename _numTp>
int read_MAT_Variables(std::string inputfile,
        std::unordered_map<std::string, matvar_t*> &matlab_readvar_map,
        SAR_Aperture<_numTp>& aperture) {
    mat_t *matfp = Mat_Open(inputfile.c_str(), MAT_ACC_RDONLY);
    std::vector<std::string> context;
    if (matfp) {
        matvar_t* root_matVar;
        while ((root_matVar = Mat_VarReadNext(matfp)) != NULL) {
            std::string varName(root_matVar->name);
            if (root_matVar->data_type == matio_types::MAT_T_STRUCT) {
                context.push_back(varName);
                read_MAT_Struct(matfp, root_matVar, matlab_readvar_map, context, aperture);
            } else if (root_matVar->data_type == matio_types::MAT_T_CELL) {
                std::cout << CONCAT_STRINGVECTOR(context) + varName << "is data of type MAT_T_CELL and cannot be read." << std::endl;
            } else {
                std::string current_fieldname = CONCAT_STRINGVECTOR(context) + varName;
                std::cout << current_fieldname << " reading data..." << std::endl;
                int read_err = Mat_VarReadDataAll(matfp, root_matVar);
                if (read_err) {
                    std::cout << "Error reading " << current_fieldname << " from the MAT file." << std::endl;
                } else {
                    Mat_VarPrint(root_matVar, 1);
                    import_GOTCHA_MATData(root_matVar, current_fieldname, aperture);
                }
            }
            Mat_VarFree(root_matVar);
            root_matVar = NULL;
        }
        Mat_Close(matfp);
    } else {
        std::cout << "Could not open MATLAB file " << inputfile << "." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

template<typename _numTp>
int initializeSARFocusingVariables(SAR_Aperture<_numTp>& aperture) {
    aperture.numRangeSamples = aperture.sampleData.shape[0];
    aperture.numAzimuthSamples = aperture.sampleData.shape[1];
    aperture.numPolarities = (aperture.sampleData.shape.size() > 2) ? aperture.sampleData.shape[2] : 1;
    int numSARSamples = aperture.numRangeSamples * aperture.numAzimuthSamples;
    if (aperture.Ant_x.numValues(aperture.polarity_dimension) != aperture.numAzimuthSamples ||
            aperture.Ant_y.numValues(aperture.polarity_dimension) != aperture.numAzimuthSamples ||
            aperture.Ant_z.numValues(aperture.polarity_dimension) != aperture.numAzimuthSamples) {
        std::cout << "initializeSARFocusingVariables::Not enough antenna positions available to focus the selected SAR data." << std::endl;
    }
    if (aperture.freq.numValues(aperture.polarity_dimension) != numSARSamples) {
        std::cout << "initializeSARFocusingVariables::Found " << aperture.freq.numValues(aperture.polarity_dimension)
                << " frequency measurements and need " << numSARSamples << " measurements. Augmenting frequency data for SAR focusing." << std::endl;
        if (!aperture.freq.isEmpty() && aperture.freq.shape[0] == aperture.numRangeSamples) {
            std::cout << "Assuming constant frequency samples for each SAR pulse." << std::endl;
            // make aperture.numAzimuthSamples-1 copies of the first frequency sample vector
            aperture.freq.shape.clear();
            aperture.freq.shape.push_back(aperture.numRangeSamples);
            aperture.freq.shape.push_back(aperture.numAzimuthSamples);
            for (int azIdx = 1; azIdx < aperture.numAzimuthSamples; ++azIdx) {
                aperture.freq.data.insert(aperture.freq.data.end(), &aperture.freq.data[0], &aperture.freq.data[aperture.numRangeSamples]);
            }
        } else if (!aperture.startF.isEmpty() && aperture.startF.shape[1] == aperture.numAzimuthSamples &&
                !aperture.ADF.isEmpty() && aperture.ADF.shape[0] == 1 &&
                !aperture.chirpRate.isEmpty() && aperture.chirpRate.shape[1] == aperture.numAzimuthSamples) {
            std::cout << "Assuming variable frequency samples for each SAR pulse. Interpolating frequency samples from chirp rate, sample rate and start frequency." << std::endl;
            aperture.freq.data.clear();
            aperture.freq.shape.clear();
            aperture.freq.shape.push_back(aperture.numRangeSamples);
            aperture.freq.shape.push_back(aperture.numAzimuthSamples);
            for (int azIdx = 0; azIdx < aperture.numAzimuthSamples; ++azIdx) {
                for (int freqIdx = 0; freqIdx < aperture.numRangeSamples; freqIdx++) {
                    _numTp freqSample = aperture.startF.data[azIdx] + freqIdx * aperture.chirpRate.data[azIdx] / aperture.ADF.data[0];
                    aperture.freq.data.push_back(freqSample);
                }
            }
        }

    }
    if (aperture.slant_range.numValues(aperture.polarity_dimension) != aperture.numAzimuthSamples) {
        std::cout << "initializeSARFocusingVariables::Found " << aperture.slant_range.numValues(aperture.polarity_dimension)
                << " slant range measurements and need " << aperture.numAzimuthSamples << " measurements. Augmenting slant range data for SAR focusing." << std::endl;
        aperture.slant_range.shape.clear();
        aperture.slant_range.shape.data();
        aperture.slant_range.shape.push_back(aperture.numAzimuthSamples);
        for (int azIdx = 0; azIdx < aperture.numAzimuthSamples; ++azIdx) {
            aperture.slant_range.data.push_back(std::sqrt((aperture.Ant_x.data[azIdx] * aperture.Ant_x.data[azIdx]) +
                    (aperture.Ant_y.data[azIdx] * aperture.Ant_y.data[azIdx]) +
                    (aperture.Ant_z.data[azIdx] * aperture.Ant_z.data[azIdx])));
        }
    }
}
#endif /* CPUBACKPROJECTION_MAIN_HPP */

