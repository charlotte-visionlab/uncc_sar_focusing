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
 * File:   example_gpu.hpp
 * Author: local-arwillis
 *
 * Created on April 6, 2022, 11:05 AM
 */

#ifndef EXAMPLE_GPU_HPP
#define EXAMPLE_GPU_HPP

template<typename __nTp>
int sph_sar_data_callback_gpu(
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
}

#endif /* EXAMPLE_GPU_HPP */

