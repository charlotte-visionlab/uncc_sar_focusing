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
 * File:   uncc_sar_globals.hpp
 * Author: arwillis
 *
 * Created on April 23, 2022, 4:57 PM
 */

#ifndef UNCC_SAR_GLOBALS_HPP
#define UNCC_SAR_GLOBALS_HPP

#include <cxxopts.hpp>

#define LOADBMP_IMPLEMENTATION
#include <loadbmp.h>

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

void cxxopts_integration(cxxopts::Options& options) {

    options.add_options()
            ("i,input", "Input file", cxxopts::value<std::string>())
            //("f,format", "Data format {GOTCHA, Sandia, <auto>}", cxxopts::value<std::string>()->default_value("auto"))
            ("p,polarity", "Polarity {HH,HV,VH,VV,<any>}", cxxopts::value<std::string>()->default_value("any"))
            ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
            ("r,dynrange", "Dynamic Range (dB) <70 dB>", cxxopts::value<float>()->default_value("70"))
            ("o,output", "Output file <sar_image.bmp>", cxxopts::value<std::string>()->default_value("sar_image.bmp"))
            ("h,help", "Print usage")
            ;
}

template<typename __nTp, typename __pTp>
int writeBMPFile(const SAR_ImageFormationParameters<__pTp>& SARImgParams,
        const CArray<__nTp>& output_image, const std::string& output_filename) {

    unsigned int width = SARImgParams.N_x_pix, height = SARImgParams.N_y_pix;
    std::vector<unsigned char> pixels;
    float max_val = std::accumulate(std::begin(output_image), std::end(output_image), 0.0f,
            [](const Complex<__nTp>& a, const Complex<__nTp> & b) {
                auto abs_a = Complex<__nTp>::abs(a);
                auto abs_b = Complex<__nTp>::abs(b);
                //auto abs_a = std::abs(a);
                //auto abs_b = std::abs(b);
                if (abs_a == abs_b) {
                    //return std::max(arg(a), arg(b));
                    return abs_a;
                }
                return std::max(abs_a, abs_b);
            });

    bool flipY = false;
    bool flipX = true;
    int srcIndex;
    for (int y_dstIndex = 0; y_dstIndex < SARImgParams.N_y_pix; y_dstIndex++) {
        for (int x_dstIndex = 0; x_dstIndex < SARImgParams.N_x_pix; x_dstIndex++) {
            if (flipX && flipY) {
                srcIndex = (SARImgParams.N_x_pix - 1 - x_dstIndex) * SARImgParams.N_y_pix + SARImgParams.N_y_pix - 1 - y_dstIndex;
            } else if (flipY) {
                srcIndex = (SARImgParams.N_x_pix - 1 - x_dstIndex) * SARImgParams.N_y_pix + y_dstIndex;
            } else if (flipX) {
                srcIndex = x_dstIndex * SARImgParams.N_y_pix + SARImgParams.N_y_pix - 1 - y_dstIndex;
            } else {
                srcIndex = x_dstIndex * SARImgParams.N_y_pix + y_dstIndex;
            }
            const Complex<__nTp>& SARpixel = output_image[srcIndex];
            //float pixelf = (float) (255.0 / SARImgParams.dyn_range_dB)*
            //        ((20 * std::log10(std::abs(SARpixel) / max_val)) + SARImgParams.dyn_range_dB);
            float pixelf = (float) (255.0 / SARImgParams.dyn_range_dB)*
                    ((20 * std::log10(Complex<__nTp>::abs(SARpixel) / max_val)) + SARImgParams.dyn_range_dB);
            unsigned char pixel = (pixelf < 0) ? 0 : (unsigned char) pixelf;
            // insert 4 copies of the pixel value
            pixels.insert(pixels.end(), 4, pixel);
        }
    }

    unsigned int err = loadbmp_encode_file(output_filename.c_str(),
            &pixels[0], width, height, LOADBMP_RGBA);

    if (err) {
        std::cout << "writeBMPFile::LoadBMP error = " << err << " when saving image to file " << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

#endif /* UNCC_SAR_GLOBALS_HPP */

