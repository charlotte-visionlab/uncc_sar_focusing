function pulse_contrib = focusalg_MF(data, pulseIndex)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs a matched filter operation.  The following    %
% fields need to be populated:                                         %
%                                                                      %
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

%c = 299792458;
pulse_contrib = zeros(size(data.x_mat));

% Calculate differential range for each pixel in the image (m)
dR = sqrt((data.AntX(pulseIndex)-data.x_mat).^2 + ...
    (data.AntY(pulseIndex)-data.y_mat).^2 + ...
    (data.AntZ(pulseIndex)-data.z_mat).^2) - data.R0(pulseIndex);

% Calculate the frequency of each sample in the pulse (Hz)
freq = data.minF(pulseIndex) + (0:(data.K-1)) * data.deltaF;

% Perform the Matched Filter operation
for jj = 1:data.K
    pulse_contrib = pulse_contrib + data.phdata(jj,pulseIndex) * exp(1i*4*pi*freq(jj)/data.c*dR);
end
