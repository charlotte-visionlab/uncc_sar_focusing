/*
 * Copyright (C) 2021 Andrew R. Willis
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
 * File:   uncc_sar_matlab.hpp
 * Author: arwillis
 *
 * Created on April 24, 2022, 7:03 AM
 */

#ifndef UNCC_SAR_MATLAB_HPP
#define UNCC_SAR_MATLAB_HPP

#include <unordered_map>

#include <matio.h>

#include "uncc_sar_focusing.hpp"
#include "uncc_simplematrix.hpp"

void initialize_GOTCHA_MATRead(std::unordered_map<std::string, matvar_t*>& matlab_readvar_map);

void initialize_Sandia_SPHRead(std::unordered_map<std::string, matvar_t*> &matlab_readvar_map);

template<typename _numTp>
int import_MATVector(matvar_t* matVar, SimpleMatrix<_numTp>& sMat) {
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
            return EXIT_FAILURE;            
    }
    return EXIT_SUCCESS;
}

template<typename _realTp>
int import_MATMatrixReal(matvar_t* matVar, SimpleMatrix<_realTp>& sMat) {
    int ndims = matVar->rank;
    int sizes[ndims];
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        sizes[ndims] = matVar->dims[dimIdx];
        sMat.shape.push_back(matVar->dims[dimIdx]);
        totalsize = totalsize * sizes[ndims];
    }

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
int import_MATMatrixComplex(matvar_t* matVar, SimpleMatrix<_complexTp>& sMat) {
    int ndims = matVar->rank;
    //int sizes[ndims];
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        //sizes[ndims] = matVar->dims[dimIdx];
        sMat.shape.push_back(matVar->dims[dimIdx]);
        totalsize = totalsize * matVar->dims[dimIdx];
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

#endif /* UNCC_SAR_MATLAB_HPP */

