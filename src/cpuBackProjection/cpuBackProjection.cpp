#include <iomanip>
#include <limits>   // std::numeric_limits

#include "cpuBackProjection.hpp"

void run_bp(const CArray& phd, float* xObs, float* yObs, float* zObs, float* r,
        int Npulses, int Nrangebins, int Nx_pix, int Ny_pix, int Nfft,
        CArray& output_image, float* minF, float* deltaF,
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
    std::cout << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << maxWr << " m range, " 
            << maxWx << " m cross-range" << std::endl;
    std::cout << "Resolution:  " << std::fixed << std::setprecision(2) << dr << "m range, " 
            << dx << " m cross-range" << std::endl;

    /*
        % Calculate the range to every bin in the range profile (m)
        data.r_vec = linspace(-data.Nfft/2,data.Nfft/2-1,data.Nfft)*data.maxWr/data.Nfft;
     */

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
    float timeleft = 0.0f;
    for (int pulseIndex = 0; pulseIndex < Npulses; pulseIndex++) {
        if (pulseIndex > 1 && (pulseIndex % 100) == 0) {
            std::cout << "Pulse " << pulseIndex << " of " << Npulses 
                    << ", " << std::setprecision(2) << timeleft << " minutes remaining" << std::endl;            
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
    int rightIdx = std::floor(xIndex)-1;
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


