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

// this declaration needs to be in any C++ compiled target for CPU
#define CUDAFUNCTION

#include "uncc_sar_matio.hpp"

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
    //    matlab_readvar_map["sph_MATData.Data.ChirpRateDelta"] = NULL;
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