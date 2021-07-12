function pulse_contrib = focusalg_PF(data, pulseIndex)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs a basic polar formatting operation.  The      %
% following fields need to be populated:                               %
%                                                                      %
% data.Nfft:  Size of the FFT to form the range profile                %
% data.deltaF:  Step size of frequency data (Hz)                       %
% data.minF:  Vector containing the start frequency of each pulse (Hz) %
% data.x_mat:  The x-position of each pixel (m)                        %
% data.y_mat:  The y-position of each pixel (m)                        %
% data.z_mat:  The z-position of each pixel (m)                        %
% data.AntX:  The x-position of the sensor at each pulse (m)           %
% data.AntY:  The y-position of the sensor at each pulse (m)           %
% data.AntZ:  The z-position of the sensor at each pulse (m)           %
% data.R0:  The range to scene center (m)                              %
% data.phdata:  Phase history data (frequency domain)                  %
%               Fast time in rows, slow time in columns                %
% data.freq_scales: The frequency scaling for this pulse               %
%                                                                      %
% The output is:                                                       %
% pulse_contrib:  The contribution of this pulse to the final          %
%                 complex image value at each pixel                    %
% Written by Andrew Willis, Univ. of Florida	                       %
% Email:  andrewwillis@ufl.edu                                         %
% Date Released:  2 June 2021                                          %
%                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = 299792458;
pulse_contrib = zeros(size(data.x_mat));

% Form the range profile with zero padding added
numPulses = size(data.phdata,2);
data.phdata(:, pulseIndex) = data.phdata(:, pulseIndex).*exp((-2j*pi/numPulses)*data.freq_scales);
rc = fftshift(ifft(data.phdata(:,pulseIndex),data.Nfft));
freq_vec = data.minF(pulseIndex):data.deltaF:(data.minF(pulseIndex)+(data.K-1)*data.deltaF);
pulseDuration = 2*data.dr/c;

% Calculate differential range for each pixel in the image (m)
dR = sqrt((data.AntX(pulseIndex)-data.x_mat).^2 + ...
    (data.AntY(pulseIndex)-data.y_mat).^2 + ...
    (data.AntZ(pulseIndex)-data.z_mat).^2) - data.R0(pulseIndex);

% Calculate phase correction for image
phCorr = exp(1i*4*pi*data.minF(pulseIndex)*dR/c);

% Calculate the range to every bin in the range profile (m)
data.r_vec = linspace(-data.Nfft/2,data.Nfft/2-1,data.Nfft)*data.maxWr/data.Nfft;
% Determine which pixels fall within the range swath
I = find(and(dR > min(data.r_vec), dR < max(data.r_vec)));

% Update the image using linear interpolation
pulse_contrib(I) = interp1(data.r_vec, rc, dR(I), 'linear') .* phCorr(I);



