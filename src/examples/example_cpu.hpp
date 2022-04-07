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
 * File:   example_cpu.hpp
 * Author: local-arwillis
 *
 * Created on April 6, 2022, 11:05 AM
 */

#ifndef EXAMPLE_CPU_HPP
#define EXAMPLE_CPU_HPP

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

template<typename __nTp>
int sph_sar_data_callback_cpu(
        int sph_MATData_total_pulses,
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_ChirpRate,
        Complex<__nTp> *sph_MATData_Data_SampleData, // ask Derek here
        int numSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numPulses) {
    SAR_Aperture<NumericType> SAR_aperture_data;
    //if (read_MAT_Variables(inputfile, matlab_readvar_map, SAR_aperture_data) == EXIT_FAILURE) {
}

#endif /* EXAMPLE_CPU_HPP */

