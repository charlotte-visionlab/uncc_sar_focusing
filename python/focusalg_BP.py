import numpy as np
from scipy.interpolate import interp1d
def focusalg_BP(x_mat, y_mat, z_mat, phdata, pulseIndex, Nfft, AntX, AntY, AntZ, minF, maxWr, R0):
    c = 299792458.0
    pulse_contrib = np.zeros(x_mat.shape)

    nfreq = phdata.shape[0]
    filter = np.ones((nfreq,1))
    pctRemoved = 0
    ndelFreqIdxs = np.round(nfreq*pctRemoved/100)
    ndelFreqIdxs = ndelFreqIdxs + np.mod(ndelFreqIdxs, 2)
    zeroFreqIdxs = np.floor(nfreq/2) + np.arange(((-ndelFreqIdxs/2)+np.mod(nfreq+1,2)),(ndelFreqIdxs/2))
    # filter[zeroFreqIdxs][0] = 0
    # phdata[:][pulseIndex] = phdata[:][pulseIndex]*filter
    # Form the range profile with zero padding added
    rc = np.fft.fftshift(np.fft.fft(phdata[:][pulseIndex], Nfft))

    # Calculate differential range for each pixel in the image (m)
    dR = np.sqrt(( AntX[0][pulseIndex] - x_mat)**2 + ( AntY[0][pulseIndex]- y_mat)**2 + ( AntZ[0][pulseIndex]- z_mat)**2) -  R0[0][pulseIndex]

    # Calculate phase correction for image
    phCorr = np.exp(1j*4*np.pi* minF*dR/ c)

    # Calculate the range to every bin in the range profile (m)
    r_vec = np.linspace(- Nfft/2, Nfft/2-1, Nfft)* maxWr/ Nfft
    # Determine which pixels fall within the range swath
    I = np.where(np.logical_and(dR > min( r_vec), dR < max( r_vec)))

    # Update the image using linear interpolation
    set_interp = interp1d(r_vec, rc)
    pulse_contrib[I] = set_interp(dR[I]) * phCorr[I]

    return pulse_contrib