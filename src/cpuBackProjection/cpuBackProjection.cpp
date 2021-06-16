
#include <stdio.h>  // printf 
#include <limits>   // std::numeric_limits
#include <time.h>

#include "cpuBackProjection.hpp"

#define PI2 6.2831853071800f   /* 2*pi */
#define PI_4__CLIGHT (4.0f * PI / CLIGHT)

#define ASSUME_Z_0    1     /* Ignore consult_DEM() and assume height = 0. */
#define USE_FAST_MATH 0     /* Use __math() functions? */
#define USE_RSQRT     0

#define MEXDEBUG      1

#define FLOAT_CLASS   mxSINGLE_CLASS

#ifndef VERBOSE
#define VERBOSE       0
#endif

/***
 * Type defs
 * ***/
typedef float FloatType; /* FIXME: this should be used everywhere */

void convert_f0(float* vec, int N) {
    int i;
    for (i = 0; i < N; ++i)
        vec[i] *= PI_4__CLIGHT;
}

float extract_f0(float* vec, int N) {
    /* Mean ...
    int i;
    float sum = 0;
    for (i=0; i<N; ++i) {
        sum += vec[i];
    }
    return sum / N;
     */
    return vec[0];
}

void fft(CArray& x);
void ifft(CArray& x);
void fftw(CArray& x);
void ifftw(CArray& x);

CArray fftshift(CArray& fft);

void run_bp(const CArray& phd, float* xObs, float* yObs, float* zObs, float* r,
        int Npulses, int Nrangebins, int Nx_pix, int Ny_pix, int Nfft,
        int blockwidth, int blockheight,
        int deviceId, CArray& output_image,
        int start_output_index, int num_output_rows,
        float c__4_delta_freq, float pi_4_f0__clight, float* minF, float* deltaF,
        float x0, float y0, float Wx, float Wy,
        float min_eff_idx, float total_proj_length) {

    /*
        % Determine the azimuth angles of the image pulses (radians)
        data.AntAz = unwrap(atan2(data.AntY,data.AntX));
        % Determine the average azimuth angle step size (radians)
        data.deltaAz = abs(mean(diff(data.AntAz)));
        % Determine the total azimuth angle of the aperture (radians)
        data.totalAz = max(data.AntAz) - min(data.AntAz);
     */

    float AntAz[Npulses];
    float deltaAz[Npulses - 1];
    float minAz = std::numeric_limits<float>::infinity();
    float maxAz = -std::numeric_limits<float>::infinity();
    float meanDeltaAz = 0.0f;
    float meanMinF = 0.0f;
    for (int pulseIndex = 0; pulseIndex < Npulses; pulseIndex++) {
        // TODO: we are not unwrapping the phase here
        //unwrap(atan2(data.AntY,data.AntX));
        AntAz[pulseIndex] = std::atan2(yObs[pulseIndex], xObs[pulseIndex]); 
        if (pulseIndex > 0) {
            deltaAz[pulseIndex - 1] = AntAz[pulseIndex] - AntAz[pulseIndex - 1];
            meanDeltaAz += std::abs(deltaAz[pulseIndex - 1]);
        }
        if (minAz > AntAz[pulseIndex]) {
            minAz = AntAz[pulseIndex];
        }
        if (maxAz < AntAz[pulseIndex]) {
            maxAz = AntAz[pulseIndex];
        }
        meanMinF += minF[pulseIndex];
    }
    meanDeltaAz /= (float) (Npulses - 1);
    meanMinF /= (float) (Npulses);
    float totalAz = maxAz - minAz;
    /*
        % Determine the maximum scene size of the image (m)
        data.maxWr = c/(2*data.deltaF);   
        data.maxWx = c/(2*data.deltaAz*mean(data.minF));
     */
    float maxWr = (float) (CLIGHT / (2.0 * (double) deltaF[0]));
    float maxWx = (float) (CLIGHT / (2.0 * meanDeltaAz * meanMinF));
    /*
        % Determine the resolution of the image (m)
        data.dr = c/(2*data.deltaF*data.K);
        data.dx = c/(2*data.totalAz*mean(data.minF));
     */
    float dr = (float) (CLIGHT / (2.0f * deltaF[0] * Nrangebins));
    float dx = (float) (CLIGHT / (2.0f * totalAz * meanMinF));
    /*
    % Display maximum scene size and resolution
     */
    printf("Maximum Scene Size:  %.2f m range, %.2f m cross-range\n", maxWr, maxWx);
    printf("Resolution:  %.2fm range, %.2f m cross-range\n", dr, dx);
    /*
        % Calculate the range to every bin in the range profile (m)
        data.r_vec = linspace(-data.Nfft/2,data.Nfft/2-1,data.Nfft)*data.maxWr/data.Nfft;
     */
// idx should be integer    
#define RANGE_INDEX_TO_RANGE_VALUE(idx, maxWr, N) ((float) idx / N - 0.5f) * maxWr
// val should be float
#define RANGE_VALUE_TO_RANGE_INDEX(val, maxWr, N) (val / maxWr + 0.5f) * N

    float r_vec[Nfft];
    float min_Rvec = std::numeric_limits<float>::infinity();
    float max_Rvec = -std::numeric_limits<float>::infinity();
    for (int rIdx = 0; rIdx < Nfft; rIdx++) {
        // -maxWr/2:maxWr/Nfft:maxWr/2
        //float rVal = ((float) rIdx / Nfft - 0.5f) * maxWr;
        float rVal = RANGE_INDEX_TO_RANGE_VALUE(rIdx, maxWr, Nfft);
        r_vec[rIdx] = rVal;
        if (min_Rvec > r_vec[rIdx]) {
            min_Rvec = r_vec[rIdx];
        }
        if (max_Rvec < r_vec[rIdx]) {
            max_Rvec = r_vec[rIdx];
        }
    }

    for (int pulseIndex = 0; pulseIndex < Npulses; pulseIndex++) {
        if (pulseIndex > 1 && (pulseIndex % 100) == 0) {
            printf("Pulse %d of %d, %.02f minutes remaining\n", pulseIndex, Npulses, 0.0f);
        }

        CArray phaseData = phd[std::slice(pulseIndex * Nfft, Nfft, 1)];

        //ifft(phaseData);
        ifftw(phaseData);

        CArray rangeCompressed = fftshift(phaseData);

        computeDifferentialRangeAndPhaseCorrections(xObs, yObs, zObs,
                r, pulseIndex, minF,
                Npulses, Nrangebins, Nx_pix, Ny_pix, Nfft,
                x0, y0, Wx, Wy,
                r_vec, rangeCompressed, min_Rvec, max_Rvec, maxWr,
                output_image);

    }
}

// Vq = interp1(X,V,Xq) interpolates to find Vq, the value of the
// underlying function Vq=f(Xq) at the query points Xq given
// measurements of the function at X=f(V).

Complex interp1(const float* xSampleLocations, const int nSamples, const CArray& sampleValues, const float xInterpLocation, const float xIndex) {
    Complex iVal(0, 0);
    int rightIdx = std::floor(xIndex);
    while (++rightIdx < nSamples && xSampleLocations[rightIdx] <= xInterpLocation);    
    if (rightIdx == nSamples || rightIdx == 0) {
        std::cout << "Error::Invalid interpolation range." << std::endl;
        return iVal;
    }
    //if (rightIdx < (int) std::floor(xIndex)) {
    //    std::cout << "Error incorrect predicted location for dR. rightIdx = " << rightIdx << " dR_Idx= " << std::ceil(xIndex) << std::endl;
    //}
    float alpha = (xInterpLocation - xSampleLocations[rightIdx - 1]) / (xSampleLocations[rightIdx] - xSampleLocations[rightIdx - 1]);
    iVal = alpha * sampleValues[rightIdx] + (1.0f - alpha) * sampleValues[rightIdx - 1];
    return iVal;
}

void computeDifferentialRangeAndPhaseCorrections(const float* xObs, const float* yObs, const float* zObs,
        const float* range_to_phasectr, const int pulseIndex, const float* minF,
        const int Npulses, const int Nrangebins, const int Nx_pix, const int Ny_pix, int Nfft,
        const float x0, const float y0, const float Wx, const float Wy,
        const float* r_vec, const CArray& rangeCompressed, 
        const float min_Rvec, const float max_Rvec, const float maxWr,
        CArray& output_image) {
    float4 target;
    target.x = x0 - (Wx / 2);
    target.z = 0;
    float delta_x = Wx / (Nx_pix - 1);
    float delta_y = Wy / (Ny_pix - 1);
    //std::cout << "(minRvec,maxRvec) = (" << min_Rvec << ", " << max_Rvec << ")" << std::endl;
    for (int xIdx = 0; xIdx < Nx_pix; xIdx++) {
        target.y = y0 - (Wy / 2);
        for (int yIdx = 0; yIdx < Ny_pix; yIdx++) {
            float dR_val = std::sqrt((xObs[pulseIndex] - target.x) * (xObs[pulseIndex] - target.x) +
                    (yObs[pulseIndex] - target.y) * (yObs[pulseIndex] - target.y) +
                    (zObs[pulseIndex] - target.z) * (zObs[pulseIndex] - target.z)) - range_to_phasectr[pulseIndex];
            //  std::cout << "y= " << target.y << " dR(" << xIdx << ", " << yIdx << ") = " << dR_val << std::endl;
            if (dR_val > min_Rvec && dR_val < max_Rvec) {
                Complex phCorr_val = polarToComplex(1.0f, (float) ((4.0 * PI * minF[pulseIndex] * dR_val) / CLIGHT));
                //std::cout << "idx = " << (xIdx * Ny_pix + yIdx) << " (x,y)=(" << target.x << "," << target.y << ")"
                //        << "(dR,phCorr)=(" << dR_val << ", " << phCorr_val << ")" << std::endl;
                float dR_idx = RANGE_VALUE_TO_RANGE_INDEX(dR_val, maxWr, Nfft);
                Complex iRC_val = interp1(r_vec, Nfft, rangeCompressed, dR_val, dR_idx);
                int outputIdx = xIdx * Ny_pix + yIdx;
                //std::cout << "output[" << outputIdx << "] += " << (iRC_val * phCorr_val) << std::endl;
                output_image[xIdx * Ny_pix + yIdx] += iRC_val * phCorr_val;
            }
            target.y += delta_y;
        }
        target.x += delta_x;
    }
}

int main(int argc, char **argv) {
    Complex test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    Complex out[8];
    CArray data(test, 8);

    // forward fft
    //fft(data);
    fftw(data);
    int N = 8;
    std::cout << "st " << sizeof (test) << std::endl;

    // http://www.fftw.org/fftw3_doc/Complex-numbers.html
    // Structure must be only two numbers in the order real, imag
    // to be binary compatible with the C99 complex type

    std::cout << "fft" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << data[i] << std::endl;
    }

    // inverse fft
    //ifft(data);
    ifftw(data);

    std::cout << std::endl << "ifft" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << data[i] << std::endl;
    }
    return EXIT_SUCCESS;
}

/* Main kernel.
 *
 * Tuning options:
 * - is it worth #defining radar parameters like start_frequency?
 *      ............  or imaging parameters like xmin/ymax?
 * - Make sure (4 pi / c) is computed at compile time!
 * - Use 24-bit integer multiplications!
 *
 * */

//void backprojection_loop(float2* full_image,
//        int Npulses, int Ny_pix, float delta_x_m_per_pix, float delta_y_m_per_pix,
//        int PROJ_LENGTH,
//        int X_OFFSET, int Y_OFFSET,
//        float C__4_DELTA_FREQ, float* PI_4_F0__CLIGHT,
//        float left, float bottom, float min_eff_idx, float4 * platform_info,
//        float * debug_effective_idx, float * debug_2, float * x_mat, float * y_mat,
//        float rmin, float rmax) {
//
//
//}

//void from_gpu_complex_to_bp_complex_split(float2 * data, bp_complex_split out, int size) {
//    int i;
//    for (i = 0; i < size; i++) {
//        out.real[i] = data[i].x;
//        out.imag[i] = data[i].y;
//    }
//}
//
//float2* format_complex_to_columns(bp_complex_split a, int width_orig, int height_orig) {
//    float2* out = (float2*) malloc(width_orig * height_orig * sizeof (float2));
//    int i, j;
//    for (i = 0; i < height_orig; i++) {
//        int origOffset = i * width_orig;
//        for (j = 0; j < width_orig; j++) {
//            int newOffset = j * height_orig;
//            out[newOffset + i].x = a.real[origOffset + j];
//            out[newOffset + i].y = a.imag[origOffset + j];
//        }
//    }
//    return out;
//}
//
//float2* format_complex(bp_complex_split a, int size) {
//    float2* out = (float2*) malloc(size * sizeof (float2));
//    int i;
//    for (i = 0; i < size; i++) {
//        out[i].x = a.real[i];
//        out[i].y = a.imag[i];
//    }
//    return out;
//}
//
//float4* format_x_y_z_r(float * x, float * y, float * z, float * r, int size) {
//    float4* out = (float4*) malloc(size * sizeof (float4));
//    int i;
//    for (i = 0; i < size; i++) {
//        out[i].x = x[i];
//        out[i].y = y[i];
//        out[i].z = z[i];
//        out[i].w = r[i];
//    }
//    return out;
//}

/* Credits: from BackProjectionKernal.c: "originally by reinke".
 * Given a float X, returns float2 Y = exp(j * X).
 *
 * __device__ code is always inlined. */

//float2 expjf(float in) {
//    float2 out;
//    float t, tb;
//#if USE_FAST_MATH
//    t = __tanf(in / 2.0f);
//#else
//    t = tan(in / 2.0f);
//#endif
//    tb = t * t + 1.0f;
//    out.x = (2.0f - tb) / tb; /* Real */
//    out.y = (2.0f * t) / tb; /* Imag */
//
//    return out;
//}
//
//float2 expjf_div_2(float in) {
//    float2 out;
//    float t, tb;
//    //t = __tanf(in - (float)((int)(in/(PI2)))*PI2 );
//    t = std::tan(in - PI * std::round(in / PI));
//    tb = t * t + 1.0f;
//    out.x = (2.0f - tb) / tb; /* Real */
//    out.y = (2.0f * t) / tb; /* Imag */
//    return out;
//}
