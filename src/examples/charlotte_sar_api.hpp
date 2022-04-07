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
 * File:   external_api.hpp
 * Author: local-arwillis
 *
 * Created on April 7, 2022, 10:53 AM
 */

#ifndef EXTERNAL_API_HPP
#define EXTERNAL_API_HPP

#include "example_gpu.hpp"

template<typename __nTp>
int sar_data_callback(
        int sph_MATData_total_pulses,
        __nTp sph_MATData_preamble_ADF,
        Complex<__nTp> *sph_MATData_Data_SampleData,
        int numSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numPulses) {
    
    sph_sar_data_callback_gpu<__nTp>(
            sph_MATData_total_pulses,
            sph_MATData_preamble_ADF,
            sph_MATData_Data_SampleData,
            numSamples,
            sph_MATData_Data_StartF,
            sph_MATData_Data_ChirpRate,
            sph_MATData_Data_radarCoordinateFrame_x,
            sph_MATData_Data_radarCoordinateFrame_y,
            sph_MATData_Data_radarCoordinateFrame_z,
            numPulses);
}

#endif /* EXTERNAL_API_HPP */

