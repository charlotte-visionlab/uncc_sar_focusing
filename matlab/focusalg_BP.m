function pulse_contrib = focusalg_BP(data, pulseIndex)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs a basic Backprojection operation.  The        %
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
%                                                                      %
% The output is:                                                       %
% data.im_final:  The complex image value at each pixel                %
%                                                                      %
% Written by LeRoy Gorham, Air Force Research Laboratory, WPAFB, OH    %
% Email:  leroy.gorham@wpafb.af.mil                                    %
% Date Released:  8 Apr 2010                                           %
%                                                                      %
% Gorham, L.A. and Moore, L.J., "SAR image formation toolbox for       %
%   MATLAB,"  Algorithms for Synthetic Aperture Radar Imagery XVII     %
%   7669, SPIE (2010).                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pulse_contrib = zeros(size(data.x_mat));

nfreq = size(data.phdata,1);
filter = ones(nfreq,1);
pctRemoved = 0;
ndelFreqIdxs = round(nfreq*pctRemoved/100);
ndelFreqIdxs = ndelFreqIdxs + mod(ndelFreqIdxs, 2);
zeroFreqIdxs = floor(nfreq/2) + (((-ndelFreqIdxs/2)+mod(nfreq+1,2)):(ndelFreqIdxs/2));
filter(zeroFreqIdxs,1) = 0;
data.phdata(:,pulseIndex) = data.phdata(:,pulseIndex).*filter;
% Form the range profile with zero padding added
rc = fftshift(fft(data.phdata(:,pulseIndex),data.Nfft));

% Calculate differential range for each pixel in the image (m)
dR = sqrt((data.AntX(pulseIndex)-data.x_mat).^2 + ...
    (data.AntY(pulseIndex)-data.y_mat).^2 + ...
    (data.AntZ(pulseIndex)-data.z_mat).^2) - data.R0(pulseIndex);

% Calculate phase correction for image
phCorr = exp(1i*4*pi*data.minF(pulseIndex)*dR/data.c);

% Calculate the range to every bin in the range profile (m)
data.r_vec = linspace(-data.Nfft/2,data.Nfft/2-1,data.Nfft)*data.maxWr/data.Nfft;
% Determine which pixels fall within the range swath
I = find(and(dR > min(data.r_vec), dR < max(data.r_vec)));

% Update the image using linear interpolation
pulse_contrib(I) = interp1(data.r_vec, rc, dR(I), 'linear') .* phCorr(I);

%dyn_range = 70;
%Iout = uint8((255/dyn_range)*((20*log10(abs(pulse_contrib)./...
%    max(max(abs(pulse_contrib))))) + dyn_range));
%set(0, 'CurrentFigure', self.fig);
%imshow(uint8(Iout));
%drawnow;
%pause
