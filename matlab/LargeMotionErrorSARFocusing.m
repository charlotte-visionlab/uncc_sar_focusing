clear all; close all; clc

clear;
% INPUT PARAMETERS START HERE %
addpath('../build/lib');
% Define the path to the base directory of the dataset
dvdPath = '../../../sar/GOTCHA/Gotcha-CP-All';

% Define input data parameters here
pass = 1;               % What pass to image (1-8)
pol = 'HH';             % What polarization to image (HH,HV,VH,VV)
minaz = 01;             % Minimum azimuth angle (degrees)
maxaz = 01;             % Maximum azimuth angle (degrees)
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
dyn_range = 70;         % dB of dynamic range to display

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

azimuth = atan2(data.AntY(1), data.AntX(1));
rotXY = [cos(azimuth) -sin(azimuth) 0; sin(azimuth) cos(azimuth) 0; 0 0 1];
trajectory_3D = [data.AntX; data.AntY; data.AntZ];
canonicalTraj = rotXY'*trajectory_3D;
npulses = length(data.AntX);
noise = mvnrnd([0; 0; 0], 1e-5*eye(3), npulses);
canonicalTraj = canonicalTraj + noise';
data.AntX = canonicalTraj(1,:);
data.AntY = canonicalTraj(2,:);
data.AntZ = canonicalTraj(3,:);

pulseIndex = 100;
BATCHSIZE = 25;
focus_pulseIndices = (pulseIndex - (BATCHSIZE - 1)):pulseIndex;
dataWin.Wx = data.Wx;          % Scene extent x (m)
dataWin.Wy = data.Wy;          % Scene extent y (m)
dataWin.Nfft = data.Nfft;        % Number of samples in FFT
dataWin.Nx = data.Nx;          % Number of samples in x direction
dataWin.Ny = data.Ny;          % Number of samples in y direction
dataWin.x0 = data.x0;            % Center of image scene in x direction (m)
dataWin.y0 = data.y0;            % Center of image scene in y direction (m)
dataWin.dyn_range = dyn_range;         % dB of dynamic range to display

dataWin.AntX = data.AntX(focus_pulseIndices);
dataWin.AntY = data.AntY(focus_pulseIndices);
dataWin.AntZ = data.AntZ(focus_pulseIndices);
dataWin.R0 = data.R0(focus_pulseIndices);
dataWin.minF = data.minF(focus_pulseIndices);
dataWin.Np = self.BATCHSIZE;

x = self.packUnknowns(data, pulseIndex);

minimizationErrorFunction = @(x) -calculateEntropy(x, dataWin);
%options = optimset('Display','iter','MaxIter',50,'PlotFcns',@optimplotfval);
options = optimset('Display','iter', 'MaxIter', 200, 'TolX', 1e-7, 'MaxFunEvals', 200);
%x_best = fminsearch(minimizationErrornction, x, options);
x_best = fminsearch(minimizationErrorFunction, x, options);
%x_best = x;

% optimData = self.unpackUnknowns(x_best);
% errorX = optimData.AntX' - data.AntX(focus_pulseIndices);
% errorY = optimData.AntY' - data.AntY(focus_pulseIndices);
% errorZ = optimData.AntZ' - data.AntZ(focus_pulseIndices);

if (false)
    % Call the backprojection function with the appropriate inputs
    data.phdata = single(data.phdata);
    data.minF = single(data.minF);
    data.R0 = single(data.R0);
    data.x_mat = single(data.x_mat);
    data.y_mat = single(data.y_mat);
    data.z_mat = single(data.z_mat);
    data.AntX = single(data.AntX);
    data.AntY = single(data.AntY);
    data.AntZ = single(data.AntZ);
    data.deltaF = single(data.deltaF);
    data = bpBasic(data);
elseif (false)
    data.z_vec = zeros(1,length(data.x_vec));
     data.phdata = single(data.phdata);
%     data.phdata = double(data.phdata);
%     data.minF = single(data.minF);
%     data.deltaF = single(data.deltaF);
%     data.R0 = single(data.R0);
%     data.AntX = single(data.AntX);
%     data.AntY = single(data.AntY);
%     data.AntZ = single(data.AntZ);
%     %data.Nfft = single(data.Nfft);
%     data.x_vec = single(data.x_vec);
%     data.y_vec = single(data.y_vec);
%     data.z_vec = single(data.z_vec);
%     data.x0 = single(data.x0);
%     data.y0 = single(data.y0);
%     data.Wx = single(data.Wx);
%     data.Wy = single(data.Wy);
    %data.phdata(1:10,1:2)
    data = bpBasic(data);
    %data = mfBasic(data);
    device = 'CPU';
    data.im_final2 = cpuBackProjection(data.phdata, data.freq, data.AntX, data.AntY, data.AntZ, data.R0, ...
        data.Nx, data.Ny, data.Nfft, data.x0, data.y0, data.Wx, data.Wy);
    %profile viewer;
else
    gpuDevice
    % to compile
    % mexcuda -v -I/usr/local/cuda-11.3/samples/common/inc CUDABackProjectionKernel.cu
    data.z_vec = zeros(1,length(data.x_vec));
    data.freq = single(data.freq);
    %data.phdata = double(data.phdata);
    data.minF = single(data.minF);
    data.R0 = single(data.R0);
    data.x_vec = single(data.x_vec);
    data.y_vec = single(data.y_vec);
    data.z_vec = single(data.z_vec);
    data.AntX = single(data.AntX);
    data.AntY = single(data.AntY);
    data.AntZ = single(data.AntZ);
    data.deltaF = single(data.deltaF);
    %data = bpBasic(data);
    data.im_final = zeros(data.Ny, data.Nx);
    device = 'GPU';
    tic;
    data.im_final2 = cuda_sar_focusing(data.phdata, data.freq, data.AntX, data.AntY, data.AntZ, data.R0, ...
        data.Nx, data.Ny, data.Nfft, data.x0, data.y0, data.Wx, data.Wy);
    toc;
end
% Display the image
figure

imagesc(data.x_vec,data.y_vec,20*log10(abs(data.im_final2)./...
    max(max(abs(data.im_final2)))),[-dyn_range 0]), colormap gray, axis xy image, title(strcat(device,' BP'));
%set(gca,'XTick',(data.x0-data.Wx)/2:data.Wx/5:(data.x0+data.Wx/2), ...
%    'YTick',-(data.y0-data.Wy)/2:data.Wy/5:(data.y0+data.Wy/2));
h = xlabel('x (m)');
%set(h,'FontSize',14,'FontWeight''Bold');
h = ylabel('y (m)');
%set(h,'FontSize',14,'FontWeight','Bold');
colorbar

function unknownVec = packUnknowns(data, numPulses)

end

function trajXYZ = unpackUnknowns(x)
    
end

function H = calculateEntropy(self, x, constData)
    optimData = self.unpackUnknowns(x);
end