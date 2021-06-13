/* 
 * File:   gpuBackProjectionKernel.cuh
 * Author: arwillis
 *
 * Created on June 12, 2021, 10:29 AM
 */

#ifndef CPUBACKPROJECTION_HPP
#define CPUBACKPROJECTION_HPP

#include <cmath>

#include <mex.h>    // Matlab library includes
#include <matrix.h> // Matlab mxComplexSingle struct

class mxComplexSingleClass : public mxComplexSingle {
public:

    mxComplexSingleClass() {
    }

    // needed for valarray operations like x /= x.size() 
    // when x is a std::valarray<mxComplexSingleClass>

    template <typename _Tp>
    mxComplexSingleClass(const _Tp& _real) {
        real = _real;
        imag = 0;
    }

    mxComplexSingleClass(const float& _real, const float& _imag) {
        real = _real;
        imag = _imag;
    }

    static mxComplexSingleClass conj(const mxComplexSingleClass& x) {
        mxComplexSingleClass z;
        z.real = x.real;
        z.imag = -x.imag;
        return z;
    }

    mxComplexSingleClass polar(mxComplexSingleClass& x) {
        mxComplexSingleClass z;
        z.real = norm(x);
        z.imag = std::atan2(x.imag, x.real);
        return z;
    }

    static mxComplexSingleClass polar(const float& r, const float& phi) {
        mxComplexSingleClass z;
        z.real = r * std::cos(phi);
        z.imag = r * std::sin(phi);
        return z;
    }
    
    float abs() {
        return norm(*this);
    }
    
    static float norm(const mxComplexSingleClass& z) {
        return std::sqrt(z.real * z.real + z.imag * z.imag);
    }

    template <typename _Tp>
    mxComplexSingleClass operator/(const _Tp& rhs) {
        mxComplexSingleClass z = *this;
        z /= rhs;        
        return z;
    }

    mxComplexSingleClass operator*(const mxComplexSingleClass& rhs) {
        mxComplexSingleClass z = *this;
        z *= rhs;
        return z;
    }

    mxComplexSingleClass operator+(const mxComplexSingleClass& rhs) {
        mxComplexSingleClass z = *this;
        z += rhs;
        return z;
    }

    mxComplexSingleClass operator-(const mxComplexSingleClass& rhs) {
        mxComplexSingleClass z = *this;
        z -= rhs;
        return z;
    }

    template <typename _Tp>
    mxComplexSingleClass operator=(const _Tp& other) {
        real = other;
        imag = 0;
        return *this;
    }

    mxComplexSingleClass& operator=(const float& other) {
        real = other;
        imag = 0;
        return *this;
    }

    mxComplexSingleClass& operator=(const mxComplexSingleClass& other) {
        real = other.real;
        imag = other.imag;
        return *this;
    }

    template <typename _Tp>
    mxComplexSingleClass& operator+=(const _Tp& rhs) {
        real += rhs.real;
        imag += rhs.imag;
        return *this;
    }

    template <typename _Tp>
    mxComplexSingleClass& operator-=(const _Tp& rhs) {
        real -= rhs.real;
        imag -= rhs.imag;
        return *this;
    }

    template <typename _Tp>
    mxComplexSingleClass& operator*=(const _Tp& rhs) {        
        imag *= rhs;
        real *= rhs;
        return *this;
    }

    mxComplexSingleClass& operator*=(const mxComplexSingleClass& rhs) {
        const float __r = real * rhs.real - imag * rhs.imag;
        imag = real * rhs.imag + imag * rhs.real;
        real = __r;
        return *this;
    }

    template <typename _Tp>
    mxComplexSingleClass& operator/=(const _Tp& rhs) {
        real /= rhs;
        imag /= rhs;
        return *this;
    }

    mxComplexSingleClass& operator/=(const mxComplexSingleClass& rhs) {
        const float __r = real * rhs.real + imag * rhs.imag;
        const float __n = norm(rhs);
        imag = (imag * rhs.real - real * rhs.imag) / __n;
        real = __r / __n;
        return *this;
    }
};

#define REAL(vec) (vec.x)
#define IMAG(vec) (vec.y)

#define CAREFUL_AMINUSB_SQ(x,y) __fmul_rn(__fadd_rn((x), -1.0f*(y)), __fadd_rn((x), -1.0f*(y)))

#define BLOCKWIDTH    16
#define BLOCKHEIGHT   16

#define MAKERADIUS(xpixel,ypixel, xa,ya,za) sqrtf(CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + __fmul_rn(za, za))

#define CLIGHT 299792458.0f /* c: speed of light, m/s */
#define PI 3.14159265359f   /* pi, accurate to 6th place in single precision */

class float2 {
public:
    float x, y;
};

class float4 : public float2 {
public:
    float z, w;
};

float2 expjf(float in);
float2 expjf_div_2(float in);

/* Main kernel.
 *
 * Tuning options:
 * - is it worth #defining radar parameters like start_frequency?
 *      ............  or imaging parameters like xmin/ymax?
 * - Make sure (4 pi / c) is computed at compile time!
 * - Use 24-bit integer multiplications!
 *
 * */
void backprojection_loop(float2* full_image,
        int Npulses, int Ny_pix, float delta_x_m_per_pix, float delta_y_m_per_pix,
        int PROJ_LENGTH,
        int X_OFFSET, int Y_OFFSET,
        float C__4_DELTA_FREQ, float* PI_4_F0__CLIGHT,
        float left, float bottom, float min_eff_idx, float4 * platform_info,
        float * debug_effective_idx, float * debug_2, float * x_mat, float * y_mat,
        float rmin, float rmax) {


}

/* Credits: from BackProjectionKernal.c: "originally by reinke".
 * Given a float X, returns float2 Y = exp(j * X).
 *
 * __device__ code is always inlined. */

float2 expjf(float in) {
    float2 out;
    float t, tb;
#if USE_FAST_MATH
    t = __tanf(in / 2.0f);
#else
    t = tan(in / 2.0f);
#endif
    tb = t * t + 1.0f;
    out.x = (2.0f - tb) / tb; /* Real */
    out.y = (2.0f * t) / tb; /* Imag */

    return out;
}

float2 expjf_div_2(float in) {
    float2 out;
    float t, tb;
    //t = __tanf(in - (float)((int)(in/(PI2)))*PI2 );
    t = std::tan(in - PI * std::round(in / PI));
    tb = t * t + 1.0f;
    out.x = (2.0f - tb) / tb; /* Real */
    out.y = (2.0f * t) / tb; /* Imag */
    return out;
}

#endif /* CPUBACKPROJECTION_HPP */

