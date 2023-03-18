clear;
clc;
close all;

% Define the path to the base directory of the dataset
dvdPath = '../../../sar/GOTCHA/Gotcha-CP-All';
% Define input data parameters here
pass = 1;               % What pass to image (1-8)
pol = 'HH';             % What polarization to image (HH,HV,VH,VV)
minaz = 01;             % Minimum azimuth angle (degrees)
maxaz = 05;             % Maximum azimuth angle (degrees)
af_flag = 0;            % Use autofocus flag (Only available for HH and VV)
taper_flag = 0;         % Add a hamming taper for sidelobe control

% Define image parameters here
data.Wx = 100;          % Scene extent x (m)
data.Wy = 100;          % Scene extent y (m)
data.Nfft = 424;        % Number of samples in FFT
data.Nx = 500;          % Number of samples in x direction
data.Ny = 500;          % Number of samples in y direction
data.x0 = 0;            % Center of image scene in x direction (m)
data.y0 = 0;            % Center of image scene in y direction (m)
data.dyn_range = 70;         % dB of dynamic range to display

% INPUT PARAMETERS END HERE %

% Determine data path
datadir = sprintf('%s%sDATA',dvdPath,filesep);

% Read in the data
for ii = minaz:maxaz
    % Determine file name based on input parameters
    in_fname = sprintf('%s%spass%d%s%s%sdata_3dsar_pass%d_az%03d_%s',datadir,...
        filesep,pass,filesep,pol,filesep,pass,ii,pol);
    
    % Load in the file
    newdata = load(in_fname);
    
    % If this is the first data file, define new variables to store data.
    % Otherwise, append the data file to the existing variables
    if isfield(data,'phdata')
        % Determine the number of pulses in this data file
        Nin = size(newdata.data.fp,2);
        
        % Determine the number of pulses already added
        Ncur = size(data.phdata,2);
        
        % Update the phase history
        data.phdata(:,(Ncur+1):(Ncur+Nin)) = newdata.data.fp;
        
        % Update r0, x, y, and z (all in meters)
        data.R0((Ncur+1):(Ncur+Nin)) = newdata.data.r0;
        data.AntX((Ncur+1):(Ncur+Nin)) = newdata.data.x;
        data.AntY((Ncur+1):(Ncur+Nin)) = newdata.data.y;
        data.AntZ((Ncur+1):(Ncur+Nin)) = newdata.data.z;
        
        % Update the autofocus parameters
        data.r_correct((Ncur+1):(Ncur+Nin)) = newdata.data.af.r_correct;
        data.ph_correct((Ncur+1):(Ncur+Nin)) = newdata.data.af.ph_correct;
    else
        % Create new variables for the new data
        data.phdata = newdata.data.fp;
        data.R0 = newdata.data.r0;
        data.AntX = newdata.data.x;
        data.AntY = newdata.data.y;
        data.AntZ = newdata.data.z;
        data.r_correct = newdata.data.af.r_correct;
        data.ph_correct = newdata.data.af.ph_correct;
        data.freq = newdata.data.freq;
    end
end

%zpad = (2^(ceil(log2(size(data.phdata,1)))+0)) - size(data.phdata,1)
%data.phdata = [data.phdata; zeros(zpad, size(data.phdata,2))];

% Calculate the minimum frequency for each pulse (Hz)
data.minF = min(data.freq)*ones(size(data.R0));

% Calculate the frequency step size (Hz)
data.deltaF = diff(data.freq(1:2));

if af_flag
    % r_correct is a correction applied to r0 (effectivley producing a
    % phase ramp for each pulse
    data.R0 = data.R0 + data.r_correct;
    
    % ph_correct is a phase correction applied to each sample in a pulse
    data.phdata = data.phdata .* repmat(exp(1i*data.ph_correct),[size(data.phdata,1) 1]);
end

% Determine the number of pulses and the samples per pulse
[data.K,data.Np] = size(data.phdata);

% Add a hamming taper to the data if desired
if taper_flag
    data.phdata = data.phdata .* (hamming(data.K)*hamming(data.Np)');
end

% Setup imaging grid
data.x_vec = linspace(data.x0 - data.Wx/2, data.x0 + data.Wx/2, data.Nx);
%data.x_vec(1:10)
data.y_vec = linspace(data.y0 - data.Wy/2, data.y0 + data.Wy/2, data.Ny);
%data.y_vec(1:10)
[data.x_mat,data.y_mat] = meshgrid(data.x_vec,data.y_vec);
% data.x_mat = single(data.x_mat);
% data.y_mat = single(data.y_mat);
data.z_mat = zeros(size(data.x_mat),'single');

Isar = bpBasic(data);

img = Isar.im_final;
figure(1), subplot(1,3,1), ...
    imshow(20*log10(abs(img)),[]), title('original');
rows = data.Ny;
cols = data.Nx;

%#Degrade image with random 10th order polynomial phase
linear_coeffs_actual = (rand(1,10)-0.5)*data.Nx;
x = linspace(-1,1, data.Nx);
poly = poly1d('Degree',10,'Coefficients',linear_coeffs_actual)
ph_error_actual = poly.evaluate(x');
lincoeffs = polyfit(x, ph_error_actual, 1);
line = lincoeffs(1)*x+lincoeffs(2);
ph_error_detrended = ph_error_actual'-line;
ph_error_mat = ones(rows,1)*ph_error_detrended;
img_err = fft(ifft(img , [], 2).*exp(-1j*ph_error_mat), [], 2);


figure(2), subplot(4,1,1), ...
    plot(1:data.Nx,ph_error_actual), title('actual phase error');
figure(2), subplot(4,1,2), ...
    plot(1:data.Nx,ph_error_detrended), title('actual detrended phase error');

figure(1), subplot(1,3,2), ...
    imshow(20*log10(abs(img_err)),[]), title('perturbed cross-range phase');

%#Initialize loop variables
img_af = 1.0*img_err;
max_iter = 20;
af_ph = 0;
rms = [];

%#Compute phase error and apply correction

db_down_for_window = 30;

target_window_width = 200;
win_width = min(rows-1, target_window_width);
rms = zeros(max_iter,1);

center_az_idx = ceil(size(img_af, 2)/2);

for iii=1:max_iter
    
    %#Find brightest azimuth sample in each range bin
    
    %index = np.argsort(np.abs(img_af), axis=0)[-1]
    %row_indices = vec2ind(abs(img_af));
    [tmp, maximum_along_az_idx] = max(abs(img_af), [], 2);
    %#Circularly shift image so max values line up
    f = zeros(size(img_af));
    for row=1:data.Ny
        f(row,:) = circshift(img_af(row,:), 1 + center_az_idx - maximum_along_az_idx(row));
    end
        
    % Determine window width
    noncoh_avg_window = sum(conj(img_af).*img_af, 1);
    noncoh_avg_window_dB = 10*log10(noncoh_avg_window/max(noncoh_avg_window));
    
    window_cutoff_dB = max(noncoh_avg_window_dB) - abs(db_down_for_window);
    
    leftidx  = find(noncoh_avg_window_dB(1:center_az_idx    ) - window_cutoff_dB< 0, 1, 'last' );
    rightidx = find(noncoh_avg_window_dB(center_az_idx+1:end) - window_cutoff_dB< 0, 1, 'first');
    leftidx = leftidx+1;
    rightidx = rightidx + center_az_idx - 1;
    if isempty(leftidx)
        leftidx = 1;
    end
    if isempty(rightidx)
        rightidx = data.Nx;
    end
    %leftidx=center_az_idx-25;
    %rightidx=center_az_idx+25;
    window_mask_row = zeros(1,data.Nx);
    window_mask_row(leftidx:rightidx)=1;
    window_mask_mat = ones(data.Ny,1)*window_mask_row;
    % clip off measurements outside the window
    g = f .* window_mask_mat;
    
    %figure(5), imshow(20*log10(abs(g)),[]);
    
    % Apply window, DFT resulting range lines and, estimate phase error
    %fft_length_pow_2 = 2^ceil(log2(data.Nx));
    
    g_shifted = ifftshift(g, 2);
    G = fft(g_shifted, [], 2);
    %#take derivative
    G_dot = diff(G, 1, 2);
    %phi_dot = angle( sum(  conj(G(:, 1:end-1  )) .* G(:, 2:end) , 1) );
    %#Estimate Spectrum for the derivative of the phase error
    phi_dot = sum(imag(conj(G(:,1:(end-1))).*G_dot), 1) ./ sum(abs(G(:,1:(end-1))).^2, 1);

    %#Integrate to obtain estimate of phase error
    phi_error_est = unwrap([0 cumsum(phi_dot)]);
    
    %phi_error_est = fliplr(phi_error_est);
    %#Remove linear trend
    %x = linspace(-1,1, data.Nx);
    x = 0:(data.Nx-1);
    linear_coeffs_est = polyfit(x, phi_error_est, 1);
    linear_trend = linear_coeffs_est(1)*x + linear_coeffs_est(2);
    phi_error_est_detrended = phi_error_est - linear_trend;
    %phi_error_est_detrended = phi_error_est;
    rms(iii) =  sqrt(mean(phi_error_est_detrended.^2));

    %#Store phase
    af_ph = af_ph + phi_error_est_detrended;

    figure(2), subplot(4,1,3), ...
        plot(1:data.Nx, phi_error_est), title('estimated phase error');
    figure(2), subplot(4,1,4), ...
        plot(1:data.Nx, af_ph), title('estimated detrended phase error');

    %if win == 'auto':
%             if rms[iii]<0.1:
%                 break
%         
    %#Apply correction
    phi_mat = ones(rows,1)*phi_error_est_detrended;
    IMG_af = fft(img_af, [], 2);
    IMG_af = IMG_af.*exp(-1j*phi_mat);
    img_af = ifft(IMG_af, [], 2);
    
    figure(1), subplot(1,3,3), ...
        imshow(20*log10(abs(img_af)),[]), title('recovered');
    drawnow;
    pause(1);
    aaa = 1;
end
    
% function [img_af, af_ph] = autoFocus2(img, data, win, win_params)
% ##############################################################################
% #                                                                            #
% #  This program autofocuses an image using the Phase Gradient Algorithm.     #
% #  If the parameter win is set to auto, an adaptive window is used.          #
% #  Otherwise, the user sets win to 0 and defines win_params.  The first      #
% #  element of win_params is the starting windows size.  The second element   #
% #  is the factor by which to reduce it by for each iteration.  This version  #
% #  is more suited for an image that is severely degraded (such as for the    #
% #  auto_focusing demo)                                                       #
% #  since, for the adaptive window, it uses most of the data for the first    #
% #  few iterations.  Below is the paper this algorithm is based off of.       #
% #                                                                            #
% #  D. Wahl, P. Eichel, D. Ghiglia, and J. Jakowatz, C.V., \Phase gradient    #
% #  autofocus-a robust tool for high resolution sar phase correction,"        #
% #  Aerospace and Electronic Systems, IEEE Transactions on, vol. 30,          #
% #  pp. 827{835, Jul 1994.                                                    #
% #                                                                            #
% ##############################################################################
%     
%     #Derive parameters
%     npulses = data.Np
%     nsamples = data.Nfft
%     
%     %#Initialize loop variables
%     img_af = 1.0*img
%     max_iter = 10
%     af_ph = 0
%     rms = []
%     
%     %#Compute phase error and apply correction
%     for iii=1:max_iter
%         
%         %#Find brightest azimuth sample in each range bin
%         
%         %index = np.argsort(np.abs(img_af), axis=0)[-1]
%         column_indices = vec2ind(abs(img'));
%         
%         %#Circularly shift image so max values line up   
%         f = zeros(size(img));
%         
%         
%         %for i in range(nsamples):
%         %    f[:,i] = np.roll(img_af[:,i], npulses/2-index[i])
%         
%         if win == 'auto':
%             #Compute window width    
%             s = np.sum(f*np.conj(f), axis = -1)
%             s = 10*np.log10(s/s.max())
%             #For first two iterations use all azimuth data 
%             #and half of azimuth data, respectively
%             if iii == 0:
%                 width = npulses
%             elif iii == 1:
%                 width = npulses/2
%             #For all other iterations, use twice the 30 dB threshold
%             else:
%                 width = np.sum(s>-30)
%             window = np.arange(npulses/2-width/2,npulses/2+width/2)
%         else:
%             #Compute window width using win_params if win not set to 'auto'    
%             width = int(win_params[0]*win_params[1]**iii)
%             window = np.arange(npulses/2-width/2,npulses/2+width/2)
%             if width<5:
%                 break
%         
%         #Window image
%         g = np.zeros(img.shape)+0j
%         g[window] = f[window]
%         
%         #Fourier Transform
%         G = sig.ift(g, ax=0)
%         
%         #take derivative
%         G_dot = np.diff(G, axis=0)
%         a = np.array([G_dot[-1,:]])
%         G_dot = np.append(G_dot,a,axis = 0)
%         
%         #Estimate Spectrum for the derivative of the phase error
%         phi_dot = np.sum((np.conj(G)*G_dot).imag, axis = -1)/\
%                   np.sum(np.abs(G)**2, axis = -1)
%                 
%         #Integrate to obtain estimate of phase error(Jak)
%         phi = np.cumsum(phi_dot)
%         
%         #Remove linear trend
%         t = np.arange(0,nsamples)
%         slope, intercept, r_value, p_value, std_err = linregress(t,phi)
%         line = slope*t+intercept
%         phi = phi-line
%         rms.append(np.sqrt(np.mean(phi**2)))
%         
%         if win == 'auto':
%             if rms[iii]<0.1:
%                 break
%         
%         #Apply correction
%         phi2 = np.tile(np.array([phi]).T,(1,nsamples))
%         IMG_af = sig.ift(img_af, ax=0)
%         IMG_af = IMG_af*np.exp(-1j*phi2)
%         img_af = sig.ft(IMG_af, ax=0)
%         
%         #Store phase
%         af_ph += phi    
%        
%     fig = plt.figure(figsize = (12,10))
%     ax1 = fig.add_subplot(2,2,1)
%     ax1.set_title('original')
%     ax1.imshow(10*np.log10(np.abs(img)/np.abs(img).max()), cmap = cm.Greys_r)
%     ax2 = fig.add_subplot(2,2,2)
%     ax2.set_title('autofocused')
%     ax2.imshow(10*np.log10(np.abs(img_af)/np.abs(img_af).max()), cmap = cm.Greys_r)
%     ax3 = fig.add_subplot(2,2,3)
%     ax3.set_title('rms phase error vs. iteration')
%     plt.ylabel('Phase (radians)')
%     ax3.plot(rms)
%     ax4 = fig.add_subplot(2,2,4)
%     ax4.set_title('phase error')
%     plt.ylabel('Phase (radians)')
%     ax4.plot(af_ph)
%     plt.tight_layout()
%     
% 
%     print('number of iterations: %i'%(iii+1))
%                      
%     return(img_af, af_ph)
%     