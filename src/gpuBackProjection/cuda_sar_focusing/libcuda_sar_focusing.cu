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

#define CUDAFUNCTION __host__ __device__

#include "cuda_sar_focusing.hpp"

using NumericType = float;
using ComplexType = Complex<NumericType>;
using ComplexArrayType = CArray<NumericType>;

template void cuda_focus_SAR_image<float, float>(SAR_Aperture<float> const&,
        SAR_ImageFormationParameters<float> const&,
        std::valarray<unccComplex<float> >&);

template void cuda_focus_SAR_image<double, double>(SAR_Aperture<double> const&,
                                                 SAR_ImageFormationParameters<double> const&,
                                                 std::valarray<unccComplex<double> >&);
