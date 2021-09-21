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

azimuth = atan2(data.AntY(1), data.AntX(1));
rotXY = [cos(azimuth) -sin(azimuth) 0; sin(azimuth) cos(azimuth) 0; 0 0 1];
trajectory_3D = [data.AntX; data.AntY; data.AntZ];
canonicalTraj = rotXY'*trajectory_3D;
npulses = length(data.AntX);
noise = mvnrnd([0; 0; 0], 1e-5*eye(3), npulses);
canonicalTraj = canonicalTraj + noise';
%data.AntX = canonicalTraj(1,:);
%data.AntY = canonicalTraj(2,:);
%data.AntZ = canonicalTraj(3,:);

sarImageFig = figure(1);
costFig = figure(2);

method = 'uniform';
pctPulses = 0.15;
batchsize = max(round(npulses*pctPulses), 10);
if (batchsize > data.Np)
    batchsize = data.Np;
end
Nfreq_samples = 16;
batched_data = selectPulses(data, batchsize, Nfreq_samples, method);

x = packUnknowns(batched_data);

minimizationErrorFunction = @(x) -calculateEntropy(x, batched_data);
%options = optimset('Display','iter','MaxIter',50,'PlotFcns',@optimplotfval);
options = optimset('Display','iter', 'MaxIter', 200, 'TolX', 1e-7, 'MaxFunEvals', 200);
%x_best = fminsearch(minimizationErrornction, x, options);
%x_best = fminsearch(minimizationErrorFunction, x, options);
%x_best = x;

% optimData = self.unpackUnknowns(x_best);
% errorX = optimData.AntX' - data.AntX(focus_pulseIndices);
% errorY = optimData.AntY' - data.AntY(focus_pulseIndices);
% errorZ = optimData.AntZ' - data.AntZ(focus_pulseIndices);
% 

batched_data.AntX_true = batched_data.AntX; 
batched_data.AntY_true = batched_data.AntY; 
batched_data.AntZ_true = batched_data.AntZ; 

x_vals = -1.3:0.01:1.3;
y_vals = -1.3:0.01:1.3;

% Setup error surface grid
chart_error_surf.x = 0.01;
chart_error_surf.y = 0.2;
chart_error_surf.dim_samples = 15;
chart_error_surf.x_vec = linspace(-chart_error_surf.x, chart_error_surf.x, chart_error_surf.dim_samples);
chart_error_surf.y_vec = linspace(-chart_error_surf.y, chart_error_surf.y, chart_error_surf.dim_samples);

[chart_error_surf.x_mat, chart_error_surf.y_mat] = meshgrid(chart_error_surf.x_vec, chart_error_surf.y_vec);

cost = zeros(size(chart_error_surf.x_mat));

for x_idx=1:length(chart_error_surf.x_vec)
    batched_data.AntX = batched_data.AntX_true + chart_error_surf.x_mat(1, x_idx)*normrnd(0,1, 1, batched_data.Np);
    for y_idx=1:length(chart_error_surf.y_vec)
        batched_data.AntY = batched_data.AntY_true + chart_error_surf.y_mat(y_idx,1)*normrnd(0,1, 1, batched_data.Np);
        x = packUnknowns(batched_data);
        cost(y_idx, x_idx) = minimizationErrorFunction(x);
        set(0,'CurrentFigure',costFig);
        %plot(x_vals(1:idx),cost(1:idx));
        mesh(chart_error_surf.x_mat, chart_error_surf.y_mat, cost);
        xlabel('x error std. dev');
        ylabel('y error std. dev');
        cost_titlestr = sprintf('Column Entropy Cost NumPulses = %d NumFreq = %d', batchsize, Nfreq_samples);
        title(cost_titlestr);
        drawnow;
        if (x_idx == 1 && y_idx == 1)
            cost(:,:) = cost(y_idx, x_idx)
        end
    end
end

set(0,'CurrentFigure',costFig);
xlabel('x error std. dev');
ylabel('y error std. dev');
cost_titlestr = sprintf('Column Entropy Cost NumPulses = %d NumFreq = %d', batchsize, Nfreq_samples);
title(cost_titlestr);



function batched_data = selectPulses(data, batchsize, Nfreq_samples, method)
% image formation parameters
batched_data.Wx = data.Wx;          % Scene extent x (m)
batched_data.Wy = data.Wy;          % Scene extent y (m)
batched_data.Nfft = data.Nfft;        % Number of samples in FFT
batched_data.Nx = data.Nx;          % Number of samples in x direction
batched_data.Ny = data.Ny;          % Number of samples in y direction
batched_data.x0 = data.x0;            % Center of image scene in x direction (m)
batched_data.y0 = data.y0;            % Center of image scene in y direction (m)
batched_data.dyn_range = data.dyn_range;         % dB of dynamic range to display
%batched_data.x_vec = data.x_vec;
%batched_data.y_vec = data.y_vec;
%batched_data.z_vec = zeros(1,length(data.x_vec));
Nfreqs = length(data.freq);
freqIdxs = ((Nfreqs-Nfreq_samples)/2):((Nfreqs+Nfreq_samples)/2);
batched_data.freq = data.freq(freqIdxs);
batched_data.Nfft = Nfreq_samples + 1; 
batched_data.deltaF = data.deltaF;

if (strcmpi(method,'UNIFORM'))
    datasize = length(data.AntX);
    if (batchsize > datasize)
        fprintf(1,'selectPulses() batchsize is greater than the number of data samples. Setting batchsize = datasize.\n');
        batchsize = datasize;
    end
    focus_pulseIndices = round(1:datasize/batchsize:datasize);
    % select subset of the data
    %batched_data.phdata = data.phdata(:,focus_pulseIndices);
    batched_data.phdata = data.phdata(freqIdxs,focus_pulseIndices);
    batched_data.AntX = data.AntX(focus_pulseIndices);
    batched_data.AntY = data.AntY(focus_pulseIndices);
    batched_data.AntZ = data.AntZ(focus_pulseIndices);
    batched_data.R0 = data.R0(focus_pulseIndices);
    batched_data.minF = data.minF(focus_pulseIndices);
    batched_data.Np = batchsize;
else
    errstr = sprintf('selectPulses() method %s is not a valid selection method.\n',method);
    fprintf(1,'%s',errstr);
end

end

function x = packUnknowns(data)
    x = [data.AntX, data.AntY, data.AntZ];
    x = double(x);
end

function trajXYZ = unpackUnknowns(x)
    numpts = length(x)/3;    
    trajXYZ.X = x(1:numpts);
    trajXYZ.Y = x((numpts+1):2*numpts);
    trajXYZ.Z = x((2*numpts+1):3*numpts);    
end

function H = calculateEntropy(x, constData)
    sarImageFig = figure(1);
    optimData = unpackUnknowns(x);
    data = constData;
    data.AntX = optimData.X;
    data.AntY = optimData.Y;
    data.AntZ = optimData.Z;
    imRecon = focusGPU(data);
    dyn_range = data.dyn_range;
    Iout = uint8((255/dyn_range)*((20*log10(abs(imRecon)./...
        max(max(abs(imRecon))))) + dyn_range));
    
    %set(0, 'CurrentFigure', self.fig);
    set(0,'CurrentFigure', sarImageFig);
    imshow(uint8(Iout));
    drawnow;
    
    H = 0;
    for col = 1:1:size(Iout,2)
        H = H + sum(entropy(Iout(:,col)));
    end
    Iout(Iout < 1e-6) = 0;
    H = H + 1e6*sum(Iout(Iout==0));
end

function sarImage = focusGPU(data)
    data.freq = single(data.freq);
    %data.phdata = double(data.phdata);
    data.minF = single(data.minF);
    data.R0 = single(data.R0);
    data.AntX = single(data.AntX);
    data.AntY = single(data.AntY);
    data.AntZ = single(data.AntZ);
    data.deltaF = single(data.deltaF);
    %data = bpBasic(data);
    tic;
    sarImage = cpuBackProjection(data.phdata, data.freq, data.AntX, data.AntY, data.AntZ, data.R0, ...
        data.Nx, data.Ny, data.Nfft, data.x0, data.y0, data.Wx, data.Wy);
    %sarImage = cuda_sar_focusing(data.phdata, data.freq, data.AntX, data.AntY, data.AntZ, data.R0, ...
    %    data.Nx, data.Ny, data.Nfft, data.x0, data.y0, data.Wx, data.Wy);
    toc;

end

% if (false)
%     % Call the backprojection function with the appropriate inputs
%     data.phdata = single(data.phdata);
%     data.minF = single(data.minF);
%     data.R0 = single(data.R0);
%     data.x_mat = single(data.x_mat);
%     data.y_mat = single(data.y_mat);
%     data.z_mat = single(data.z_mat);
%     data.AntX = single(data.AntX);
%     data.AntY = single(data.AntY);
%     data.AntZ = single(data.AntZ);
%     data.deltaF = single(data.deltaF);
%     data = bpBasic(data);
% elseif (false)
%     data.z_vec = zeros(1,length(data.x_vec));
%      data.phdata = single(data.phdata);
% %     data.phdata = double(data.phdata);
% %     data.minF = single(data.minF);
% %     data.deltaF = single(data.deltaF);
% %     data.R0 = single(data.R0);
% %     data.AntX = single(data.AntX);
% %     data.AntY = single(data.AntY);
% %     data.AntZ = single(data.AntZ);
% %     %data.Nfft = single(data.Nfft);
% %     data.x_vec = single(data.x_vec);
% %     data.y_vec = single(data.y_vec);
% %     data.z_vec = single(data.z_vec);
% %     data.x0 = single(data.x0);
% %     data.y0 = single(data.y0);
% %     data.Wx = single(data.Wx);
% %     data.Wy = single(data.Wy);
%     %data.phdata(1:10,1:2)
%     data = bpBasic(data);
%     %data = mfBasic(data);
%     device = 'CPU';
%     data.im_final2 = cpuBackProjection(data.phdata, data.freq, data.AntX, data.AntY, data.AntZ, data.R0, ...
%         data.Nx, data.Ny, data.Nfft, data.x0, data.y0, data.Wx, data.Wy);
%     %profile viewer;
% else
%     gpuDevice
%     % to compile
%     % mexcuda -v -I/usr/local/cuda-11.3/samples/common/inc CUDABackProjectionKernel.cu
%     data.z_vec = zeros(1,length(data.x_vec));
%     data.freq = single(data.freq);
%     %data.phdata = double(data.phdata);
%     data.minF = single(data.minF);
%     data.R0 = single(data.R0);
%     data.x_vec = single(data.x_vec);
%     data.y_vec = single(data.y_vec);
%     data.z_vec = single(data.z_vec);
%     data.AntX = single(data.AntX);
%     data.AntY = single(data.AntY);
%     data.AntZ = single(data.AntZ);
%     data.deltaF = single(data.deltaF);
%     %data = bpBasic(data);
%     data.im_final = zeros(data.Ny, data.Nx);
%     device = 'GPU';
%     tic;
%     data.im_final2 = cuda_sar_focusing(data.phdata, data.freq, data.AntX, data.AntY, data.AntZ, data.R0, ...
%         data.Nx, data.Ny, data.Nfft, data.x0, data.y0, data.Wx, data.Wy);
%     toc;
% end
% % Display the image
% figure
% 
% imagesc(data.x_vec,data.y_vec,20*log10(abs(data.im_final2)./...
%     max(max(abs(data.im_final2)))),[-dyn_range 0]), colormap gray, axis xy image, title(strcat(device,' BP'));
% %set(gca,'XTick',(data.x0-data.Wx)/2:data.Wx/5:(data.x0+data.Wx/2), ...
% %    'YTick',-(data.y0-data.Wy)/2:data.Wy/5:(data.y0+data.Wy/2));
% h = xlabel('x (m)');
% %set(h,'FontSize',14,'FontWeight''Bold');
% h = ylabel('y (m)');
% %set(h,'FontSize',14,'FontWeight','Bold');
% colorbar