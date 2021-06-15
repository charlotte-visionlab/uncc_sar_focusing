
#include <stdio.h>  // printf 
#include <limits>   // std::numeric_limits
#include <time.h>

#include <complex>
#include <cmath>
#include <valarray>
#include <vector>

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

/* From ATK imager */
typedef struct {
    float* real;
    float* imag;
} complex_split;

/* To work seamlessly with Hartley's codebase */
typedef complex_split bp_complex_split;

#ifndef NO_MATLAB

typedef mxComplexSingleClass Complex;
#define polarToComplex mxComplexSingleClass::polar
#define conjugateComplex mxComplexSingleClass::conj

#else

/*
#include <complex>
typedef std::complex<float> Complex;
#define polarToComplex std::polar
#define conjugateComplex std::conj
 */

typedef mxComplexSingleClass Complex;
#define polarToComplex mxComplexSingleClass::polar
#define conjugateComplex mxComplexSingleClass::conj


#endif

typedef std::valarray<Complex> CArray;
typedef std::vector<Complex> CVector;

/***
 * Prototypes
 * ***/

float2* format_complex_to_columns(bp_complex_split a, int width_orig,
        int height_orig);

float2* format_complex(bp_complex_split a, int size);

float4* format_x_y_z_r(float * x, float * y, float * z, float * r, int size);

void run_bp(const CArray& phd, float* xObs, float* yObs, float* zObs, float* r,
        int Npulses, int Nrangebins, int Nx_pix, int Ny_pix, int Nfft,
        int blockwidth, int blockheight,
        int deviceId, CArray& output_image,
        int start_output_index, int num_output_rows,
        float c__4_delta_freq, float pi_4_f0__clight, float* minF, float* deltaF,
        float x0, float y0, float Wx, float Wy,
        float min_eff_idx, float total_proj_length);

void computeDifferentialRangeAndPhaseCorrections(const float* xObs, const float* yObs, const float* zObs,
        const float* range_to_phasectr, const int pulseIndex, const float* minF,
        const int Npulses, const int Nrangebins, const int Nx_pix, const int Ny_pix, const int Nfft,
        const float x0, const float y0, const float Wx, const float Wy,
        const float* r_vec, const CArray& rangeCompressed, const float min_Rvec, const float max_Rvec,
        std::vector<float>& dR_vec, std::vector<Complex>& phCorr_vec,
        CArray& output_image);

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

#ifndef NO_MATLAB

/* 
 * Application parameters:
 *  - range profiles
 *
 * 
 * ATK imager gets the following:
 * - range profiles (complex)
 * - f0, vector of start frequencies, Hz
 * - r0, vector of distances from radar to center of illuminated scene, m
 * - x, y, z, vectors of radar position (x points east, y north, z up), m
 * - Nimgx, Nimgy, number of pixels in x and y
 * - deltaf, spacing of frequency vector, Hz
 * - Left, right, top, bottom, corners of the square on the ground to image
 */
void mexFunction(int nlhs, /* number of LHS (output) arguments */
        mxArray* plhs[], /* array of mxArray pointers to outputs */
        int nrhs, /* number of RHS (input) args */
        const mxArray* prhs[]) /* array of pointers to inputs*/ {
    /* Section 1. 
     * These are the variables we'll use */
    /* Subsection A: these come from Matlab and are the same as the ATK code */
    mxComplexSingle* range_profiles;
    float* minF;
    float* aimpoint_ranges;
    float* xobs;
    float* yobs;
    float* zobs;
    int Nx_pix, Ny_pix;
    float* deltaF;
    float x0, y0, Wx, Wy; // image (x,y) center and (width,height) w.r.t. target phase center

    float min_eff_idx; //, Nrangebins;

    /* Subsection B: these are computed from the matlab inputs */
    int Npulses, Nrangebins, Nfft;
    float c__4_delta_freq;
    float pi_4_f0__clight;

    /* Subsection C: these are CUDA-specific options */
    int deviceId, blockwidth, blockheight;

    /* Subsection D: these are output variables */
    mxComplexSingle* output_image;

    /* Section 2. 
     * Parse Matlab's inputs */
    /* Check that both inputs are complex*/
    if (!mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:convec:inputsNotComplex",
                "Input 0 must be complex.\n");
    }

    /* Range profile dimensions */
    Npulses = mxGetN(prhs[0]);
    Nrangebins = mxGetM(prhs[0]);

    range_profiles = mxGetComplexSingles(prhs[0]);

    //range_profiles.real = (float*) mxGetPr(prhs[0]);
    //range_profiles.imag = (float*) mxGetPi(prhs[0]);

    minF = (float*) mxGetPr(prhs[1]);
    deltaF = (float*) mxGetPr(prhs[2]);

    aimpoint_ranges = (float*) mxGetPr(prhs[3]);
    xobs = (float*) mxGetPr(prhs[4]);
    yobs = (float*) mxGetPr(prhs[5]);
    zobs = (float*) mxGetPr(prhs[6]);

    Nx_pix = (int) mxGetScalar(prhs[7]);
    Ny_pix = (int) mxGetScalar(prhs[8]);

    Nfft = (int) mxGetScalar(prhs[9]);
    x0 = (float) mxGetScalar(prhs[10]);
    y0 = (float) mxGetScalar(prhs[11]);
    Wx = (float) mxGetScalar(prhs[12]);
    Wy = (float) mxGetScalar(prhs[13]);

    /* Section 3.
     * Set up some intermediate values */


    if (nrhs == 16) {
        min_eff_idx = (float) mxGetScalar(prhs[14]);
        Nrangebins = (float) mxGetScalar(prhs[15]);
    } else {
        min_eff_idx = 0;
        //Nrangebins = Nrangebins;
    }


    /* Various collection-specific constants */

    //c__4_delta_freq = CLIGHT / (4.0f * delta_frequency);
    c__4_delta_freq = 0;
    /* FIXME: this TOTALLY prevents variable start frequency!!!! */
    //pi_4_f0__clight = PI * 4.0f * extract_f0(minF, Npulses) / CLIGHT;
    pi_4_f0__clight = 0;
    //convert_f0(minF, Npulses);

    /* Section 4.
     * Set up Matlab outputs */
    plhs[0] = mxCreateNumericMatrix(Ny_pix, Nx_pix, FLOAT_CLASS, mxCOMPLEX);
    output_image = mxGetComplexSingles(plhs[0]);
    //output_image.real = (float*) mxGetPr(plhs[0]);
    //output_image.imag = (float*) mxGetPi(plhs[0]);
    mxComplexSingleClass* range_profiles_cast = static_cast<mxComplexSingleClass*> (range_profiles);
    mxComplexSingleClass* output_image_cast = static_cast<mxComplexSingleClass*> (output_image);

    CArray range_profiles_arr(range_profiles_cast, Npulses * Nrangebins);
    //CArray range_profiles_arr(Npulses * Nrangebins);
    CArray output_image_arr(output_image_cast, Ny_pix * Nx_pix);
//    for (int i = 0; i < Npulses * Nrangebins; i++) {
//        //std::cout << "I(" << i << ") = " << output_image_arr[i] << std::endl;
//        range_profiles_arr[i] = range_profiles[i];
//    }

    /*
        std::cout << "size = " << range_profiles_arr.size() << std::endl;
        int ii;
        for (int pulseIndex = 0; pulseIndex < 2; pulseIndex++) {
            CArray phaseData = range_profiles_arr[std::slice(pulseIndex*Nfft, (pulseIndex + 1) * Nfft, 1)];
            ii = 0;
            for (auto phaseSample : phaseData) {
                std::cout << "ph " << phaseSample << std::endl;
                if (++ii == 10)
                    break;
            }
        }
     */
    run_bp(range_profiles_arr, xobs, yobs, zobs,
            aimpoint_ranges,
            Npulses, Nrangebins, Nx_pix, Ny_pix, Nfft,
            blockwidth, blockheight,
            deviceId,
            output_image_arr,
            0, Ny_pix,
            c__4_delta_freq, pi_4_f0__clight,
            minF, deltaF,
            x0, y0, Wx, Wy, min_eff_idx, Nrangebins);
    for (int i = 0; i < output_image_arr.size(); i++) {
        //std::cout << "I(" << i << ") = " << output_image_arr[i] << std::endl;
        output_image[i] = output_image_arr[i];
    }

    return;
}
#endif

void from_gpu_complex_to_bp_complex_split(float2 * data, bp_complex_split out, int size) {
    int i;
    for (i = 0; i < size; i++) {
        out.real[i] = data[i].x;
        out.imag[i] = data[i].y;
    }
}

float2* format_complex_to_columns(bp_complex_split a, int width_orig, int height_orig) {
    float2* out = (float2*) malloc(width_orig * height_orig * sizeof (float2));
    int i, j;
    for (i = 0; i < height_orig; i++) {
        int origOffset = i * width_orig;
        for (j = 0; j < width_orig; j++) {
            int newOffset = j * height_orig;
            out[newOffset + i].x = a.real[origOffset + j];
            out[newOffset + i].y = a.imag[origOffset + j];
        }
    }
    return out;
}

float2* format_complex(bp_complex_split a, int size) {
    float2* out = (float2*) malloc(size * sizeof (float2));
    int i;
    for (i = 0; i < size; i++) {
        out[i].x = a.real[i];
        out[i].y = a.imag[i];
    }
    return out;
}

float4* format_x_y_z_r(float * x, float * y, float * z, float * r, int size) {
    float4* out = (float4*) malloc(size * sizeof (float4));
    int i;
    for (i = 0; i < size; i++) {
        out[i].x = x[i];
        out[i].y = y[i];
        out[i].z = z[i];
        out[i].w = r[i];
    }
    return out;
}

void run_bp(const CArray& phd, float* xObs, float* yObs, float* zObs, float* r,
        int Npulses, int Nrangebins, int Nx_pix, int Ny_pix, int Nfft,
        int blockwidth, int blockheight,
        int deviceId, CArray& output_image,
        int start_output_index, int num_output_rows,
        float c__4_delta_freq, float pi_4_f0__clight, float* minF, float* deltaF,
        float x0, float y0, float Wx, float Wy,
        float min_eff_idx, float total_proj_length) {

    // Set up platform data texture
    //float4* platform = format_x_y_z_r(xObs, yObs, zObs, r, Npulses);

    //    for (int pulseNum = 0; pulseNum < 10; pulseNum++) {
    //        printf("pulse=%d platform(x,y,z)=(%f,%f,%f) R0=%f R=%f \n", pulseNum,
    //                platform[pulseNum].x, platform[pulseNum].y, platform[pulseNum].z, platform[pulseNum].w, 0.0f);
    //    }

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
        AntAz[pulseIndex] = std::atan2(yObs[pulseIndex], xObs[pulseIndex]); //unwrap(atan2(data.AntY,data.AntX));
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
    float maxWr = CLIGHT / (2.0f * deltaF[0]);
    float maxWx = CLIGHT / (2.0f * meanDeltaAz * meanMinF);
    /*
        % Determine the resolution of the image (m)
        data.dr = c/(2*data.deltaF*data.K);
        data.dx = c/(2*data.totalAz*mean(data.minF));
     */
    float dr = CLIGHT / (2.0f * deltaF[0] * Nrangebins);
    float dx = CLIGHT / (2.0f * deltaF[0] * Nrangebins);
    /*
    % Display maximum scene size and resolution
     */
    printf("Maximum Scene Size:  %.2f m range, %.2f m cross-range\n", maxWr, maxWx);
    printf("Resolution:  %.2fm range, %.2f m cross-range\n", dr, dx);
    /*
        % Calculate the range to every bin in the range profile (m)
        data.r_vec = linspace(-data.Nfft/2,data.Nfft/2-1,data.Nfft)*data.maxWr/data.Nfft;
     */
    float r_vec[Nfft];
    float min_Rvec = std::numeric_limits<float>::infinity();
    float max_Rvec = -std::numeric_limits<float>::infinity();
    for (int rIdx = 0; rIdx < Nfft; rIdx++) {
        // -maxWr/2:maxWr/Nfft:maxWr/2
        float rVal = ((float) rIdx / Nfft - 0.5f) * maxWr;
        r_vec[rIdx] = rVal;
        if (min_Rvec > r_vec[rIdx]) {
            min_Rvec = r_vec[rIdx];
        }
        if (max_Rvec < r_vec[rIdx]) {
            max_Rvec = r_vec[rIdx];
        }
    }

    /*
         % Initialize the image with all zero values
        data.im_final = zeros(size(data.x_mat));
        % Set up a vector to keep execution times for each pulse (sec)
        t = zeros(1,data.Np);
    % Loop through every pulse
    for ii = 1:data.Np
    
        % Display status of the imaging process
        if ii > 1 && mod(ii,100)==0
            t_sofar = sum(t(1:(ii-1)));
            t_est = (t_sofar*data.Np/(ii-1)-t_sofar)/60;
            fprintf('Pulse %d of %d, %.02f minutes remaining\n',ii,data.Np,t_est);
        end
        tic

        % Form the range profile with zero padding added
        rc = fftshift(ifft(data.phdata(:,ii),data.Nfft));

        % Calculate differential range for each pixel in the image (m)
        dR = sqrt((data.AntX(ii)-data.x_mat).^2 + ...
            (data.AntY(ii)-data.y_mat).^2 + ...
            (data.AntZ(ii)-data.z_mat).^2) - data.R0(ii);

        % Calculate phase correction for image
        phCorr = exp(1i*4*pi*data.minF(ii)*dR/c);

        % Determine which pixels fall within the range swath
        I = find(and(dR > min(data.r_vec), dR < max(data.r_vec)));

        % Update the image using linear interpolation
        data.im_final(I) = data.im_final(I) + interp1(data.r_vec,rc,dR(I),'linear') .* phCorr(I);
    
        % Determine the execution time for this pulse
        t(ii) = toc;
    end
     */
    std::vector<float> dR_vec;
    std::vector<Complex> phCorr_vec;
    int ii;
    ii = 0;
    //    for (auto rvecSamp : r_vec) {
    //        std::cout << "r_vec " << rvecSamp << std::endl;
    //        if (++ii == 10)
    //            break;
    //    }
    for (int pulseIndex = 0; pulseIndex < Npulses; pulseIndex++) {
        CArray phaseData = phd[std::slice(pulseIndex * Nfft, Nfft, 1)];

        //        if (pulseIndex > -1) {
        //            ii = 0;
        //            for (auto phaseSample : phaseData) {
        //                std::cout << "ph[" << ii++ << "] = " << phaseSample << std::endl;
        //                //                                if (++ii == 1)
        //                //                                    break;
        //            }
        //        }
        //ifft(phaseData);
        ifftw(phaseData);
        //        if (pulseIndex == 1) {
        //            ii = 0;
        //            for (auto phaseSample : phaseData) {
        //                std::cout << "ifft " << phaseSample << std::endl;
        //                if (++ii == 10)
        //                    break;
        //            }
        //        }

        CArray rangeCompressed = fftshift(phaseData);
        //        if (pulseIndex == 1) {
        //            ii = 0;
        //            for (auto range : rangeCompressed) {
        //                std::cout << "rc " << range << std::endl;
        //                if (++ii == 10)
        //                    break;
        //            }
        //        }

        computeDifferentialRangeAndPhaseCorrections(xObs, yObs, zObs,
                r, pulseIndex, minF,
                Npulses, Nrangebins, Nx_pix, Ny_pix, Nfft,
                x0, y0, Wx, Wy,
                r_vec, rangeCompressed, min_Rvec, max_Rvec,
                dR_vec, phCorr_vec, output_image);

        /*
                CArray validRanges = rangeCompressed[rangeSlices[pulseIndex]];
                if (pulseIndex == 0) {
                    ii = 0;
                    for (auto vrange : validRanges) {
                        std::cout << "dR " << vrange << std::endl;
                        if (++ii == 10)
                            break;
                    }
                }
         */
        // Vq = interp1(X,V,Xq) interpolates to find Vq, the values of the
        // underlying function V=F(X) at the query points Xq.
    }
    //free(platform);
}

Complex interp1(const float* xSampleLocations, const int nSamples, const CArray& sampleValues, const float xInterpLocation) {
    Complex iVal(0, 0);
    int rightIdx = 0;
    while (++rightIdx < nSamples && xSampleLocations[rightIdx] <= xInterpLocation);
    if (rightIdx == nSamples || rightIdx == 0) {
        std::cout << "Error::Invalid interpolation range." << std::endl;
        return iVal;
    }
    float alpha = (xInterpLocation - xSampleLocations[rightIdx - 1]) / (xSampleLocations[rightIdx] - xSampleLocations[rightIdx - 1]);
    iVal = alpha * sampleValues[rightIdx] + (1.0f - alpha) * sampleValues[rightIdx - 1];
    return iVal;
}

void computeDifferentialRangeAndPhaseCorrections(const float* xObs, const float* yObs, const float* zObs,
        const float* range_to_phasectr, const int pulseIndex, const float* minF,
        const int Npulses, const int Nrangebins, const int Nx_pix, const int Ny_pix, int Nfft,
        const float x0, const float y0, const float Wx, const float Wy,
        const float* r_vec, const CArray& rangeCompressed, const float min_Rvec, const float max_Rvec,
        std::vector<float>& dR_vec, std::vector<Complex>& phCorr_vec,
        CArray& output_image) {
    float4 target;
    target.x = x0 - (Wx / 2);
    target.z = 0;
    float delta_x = Wx / (Nx_pix - 1);
    float delta_y = Wy / (Ny_pix - 1);
    //int rvecIdx_start, rvecIdx_end;
    //std::cout << "(minRvec,maxRvec) = (" << min_Rvec << ", " << max_Rvec << ")" << std::endl;
    for (int xIdx = 0; xIdx < Nx_pix; xIdx++) {
        target.y = y0 - (Wy / 2);
        //rvecIdx_start = -1;
        //rvecIdx_end = -1;
        for (int yIdx = 0; yIdx < Ny_pix; yIdx++) {
            float dR_val = std::sqrt((xObs[pulseIndex] - target.x) * (xObs[pulseIndex] - target.x) +
                    (yObs[pulseIndex] - target.y) * (yObs[pulseIndex] - target.y) +
                    (zObs[pulseIndex] - target.z) * (zObs[pulseIndex] - target.z)) - range_to_phasectr[pulseIndex];
            //  std::cout << "y= " << target.y << " dR(" << xIdx << ", " << yIdx << ") = " << dR_val << std::endl;
            if (dR_val > min_Rvec && dR_val < max_Rvec) {
                Complex phCorr_val = polarToComplex(1.0f, (4.0f * PI * minF[pulseIndex] * dR_val) / CLIGHT);
                //dR_vec.push_back(dR_val);
                //phCorr_vec.push_back(phCorr_val);
                //std::cout << "idx = " << (xIdx * Ny_pix + yIdx) << " (x,y)=(" << target.x << "," << target.y << ")"
                //        << "(dR,phCorr)=(" << dR_val << ", " << phCorr_val << ")" << std::endl;
                Complex iRC_val = interp1(r_vec, Nfft, rangeCompressed, dR_val);
                int outputIdx = xIdx * Ny_pix + yIdx;
                //std::cout << "output[" << outputIdx << "] += " << (iRC_val * phCorr_val) << std::endl;
                output_image[xIdx * Ny_pix + yIdx] += iRC_val * phCorr_val;
            }
            target.y += delta_y;
        }
        target.x += delta_x;
    }
}


#include <fftw3.h>

void fftw_engine(CArray& x, int DIR) {
    // http://www.fftw.org/fftw3_doc/Complex-numbers.html
    // Structure must be only two numbers in the order real, imag
    // to be binary compatible with the C99 complex type
    //
    // TODO: Mangle function invocations and linking based on float/double selection
    // all calls change to fftw_plan for type double complex numbers
    // and the program must then link against fftw3 not fftw3f
    fftwf_plan p = fftwf_plan_dft_1d(x.size(),
            reinterpret_cast<fftwf_complex*> (&x[0]),
            reinterpret_cast<fftwf_complex*> (&x[0]),
            DIR, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    if (DIR == FFTW_BACKWARD) {
        x /= x.size();
    }
}

void fftw(CArray& x) {
    fftw_engine(x, FFTW_FORWARD);
}

void ifftw(CArray& x) {
    fftw_engine(x, FFTW_BACKWARD);
}

CArray fftshift(CArray& fft) {
    return fft.cshift((fft.size() + 1) / 2);
}

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive

void fft_alt(CArray& x) {
    const size_t N = x.size();
    if (N <= 1) return;

    // divide
    CArray even = x[std::slice(0, N / 2, 2)];
    CArray odd = x[std::slice(1, N / 2, 2)];

    // conquer
    fft_alt(even);
    fft_alt(odd);

    // combine
    for (size_t k = 0; k < N / 2; ++k) {
        //Complex t = Complex::polar(1.0f, -2.0f * PI * k / N) * odd[k];
        Complex t = polarToComplex(1.0f, -2.0f * PI * k / N) * odd[k];
        x[k ] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

// Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
// Better optimized but less intuitive
// !!! Warning : in some cases this code make result different from not optimized version above (need to fix bug)
// The bug is now fixed @2017/05/30 

void fft(CArray &x) {
    // DFT
    unsigned int N = x.size(), k = N, n;
    double thetaT = 3.14159265358979323846264338328L / N;
    Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
    while (k > 1) {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        T = 1.0L;
        for (unsigned int l = 0; l < k; l++) {
            for (unsigned int a = l; a < N; a += n) {
                unsigned int b = a + k;
                Complex t = x[a] - x[b];
                x[a] += x[b];
                x[b] = t * T;
            }
            T *= phiT;
        }
    }
    // Decimate
    unsigned int m = (unsigned int) log2(N);
    for (unsigned int a = 0; a < N; a++) {
        unsigned int b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        if (b > a) {
            Complex t = x[a];
            x[a] = x[b];
            x[b] = t;
        }
    }
    //// Normalize (This section make it not working correctly)
    //Complex f = 1.0 / sqrt(N);
    //for (unsigned int i = 0; i < N; i++)
    //	x[i] *= f;
}

void ifft(CArray& x) {
    // conjugate the complex numbers
    //x = x.apply(mxComplexSingleClass::conj);
    x = x.apply(conjugateComplex);

    // forward fft
    fft(x);

    // conjugate the complex numbers again
    //x = x.apply(mxComplexSingleClass::conj);
    x = x.apply(conjugateComplex);

    // scale the numbers
    x /= x.size();
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