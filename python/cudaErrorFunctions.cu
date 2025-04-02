#include <cuda_runtime.h>
/*
# Need for getting data values
#   sampleData
#   numRangeSamples
#   numAzimuthSamples
#   delta_x_m_per_pix
#   delta_y_m_per_pix
#   left
#   bottom
#   rmin
#   rmax
#   Ant_x
#   Ant_y
#   Ant_z
#   slant_range
#   startF
#   sar_image_params
#   #   N_x_pix
#   #   N_y_pix
#   #   N_fft
#   #   max_Wy_m
#   #   dyn_range_dB
#   #   Wx_m*
#   #   Wy_m*
#   #   x0_m*
#   #   y0_m*
#   range_vec
#
# *Not needed for focusing image
*/

extern "C" {

// CUDA Kernel for dft Values
__global__ void dft2DCalculation_errorfunction_kernel(const float* sampleData, int numRangeSamples, int numAzimuthSamples, const float* delta_x_m_per_pix, const float* delta_y_m_per_pix, const float* left, const float* bottom, const float* rmin, const float* rmax,
    const float* Ant_x, const float* Ant_y, const float* Ant_z, const float* slant_range, const float* startF, int N_x_pix, int N_y_pix, int N_fft, const float* max_Wy_m, const float* range_vec) {

    return;
}

float errorfunc(const float* sampleData, int numRangeSamples, int numAzimuthSamples, const float* delta_x_m_per_pix, const float* delta_y_m_per_pix, const float* left, const float* bottom, const float* rmin, const float* rmax,
    const float* Ant_x, const float* Ant_y, const float* Ant_z, const float* slant_range, const float* startF, int N_x_pix, int N_y_pix, int N_fft, const float* max_Wy_m, const float* range_vec) {

    return 1;
}

// CUDA Kernel for focusing SAR image
__global__ void focus_sar_image_kernel(const float* sampleData, int numRangeSamples, int numAzimuthSamples, const float* delta_x_m_per_pix, const float* delta_y_m_per_pix, const float* left, const float* bottom, const float* rmin, const float* rmax,
const float* Ant_x, const float* Ant_y, const float* Ant_z, const float* slant_range, const float* startF, int N_x_pix, int N_y_pix, int N_fft, const float* max_Wy_m, const float* range_vec, const float* output_image) {
    return;
}

void focus_sar_image(const float* sampleData, int numRangeSamples, int numAzimuthSamples, const float* delta_x_m_per_pix, const float* delta_y_m_per_pix, const float* left, const float* bottom, const float* rmin, const float* rmax,
    const float* Ant_x, const float* Ant_y, const float* Ant_z, const float* slant_range, const float* startF, int N_x_pix, int N_y_pix, int N_fft, const float* max_Wy_m, const float* range_vec, const float* output_image) {
    return;
}

}