clear all; 
%close all;
clc;

% INPUT PARAMETERS START HERE %
% algorithms to run
algorithm(1).name = "Matched Filter";
algorithm(1).func = @focusalg_MF;
algorithm(2).name = "Backprojection";
algorithm(2).func = @focusalg_BP;
algorithm(3).name = "Polar Formatting";
algorithm(3).func = @focusalg_PF;
sarObj = SAR_LargeMotionError();
algorithm(4).name = "Backprojection w. Motion Error";
algorithm(4).func = @sarObj.focusalg_BP;

% Define the path to the base directory of the dataset
%dvdPath = '/ssip1/GOTCHA/PublicReleaseDataset3DSAR/DVD';
data_source = 'Sandia';
data.Wx = 700;          % Scene extent x (m)
data.Wy = 700;          % Scene extent y (m)
% data_source = 'GOTCHA';
% data.Wx = 100;          % Scene extent x (m)
% data.Wy = 100;          % Scene extent y (m)

%ALGORITHM_RUNLIST = struct('algorithm_func','name','noise', cell(size(ALGORITHM_ARR)));
ALGORITHM_RUNLIST(1).algorithm_func = algorithm(2).func;
%ALGORITHM_RUNLIST(1).pct_noise = 5.0;
ALGORITHM_RUNLIST(1).pct_noise = 0.0;
ALGORITHM_RUNLIST(1).name = algorithm(2).name;
% ALGORITHM_RUNLIST(2).algorithm_func = algorithm(2).func;
% ALGORITHM_RUNLIST(2).pct_noise = 10.0;
% ALGORITHM_RUNLIST(2).name = algorithm(2).name;
% ALGORITHM_RUNLIST(3).algorithm_func = algorithm(2).func;
% ALGORITHM_RUNLIST(3).pct_noise = 15.0;
% ALGORITHM_RUNLIST(3).name = algorithm(2).name;
% ALGORITHM_RUNLIST(1).algorithm_func = algorithm(4).func;
% ALGORITHM_RUNLIST(1).pct_noise = [0, 0, 0];
% ALGORITHM_RUNLIST(1).name = sprintf('%s %g% noise', algorithm(2).name, ALGORITHM_RUNLIST(1).pct_noise(3)*100);

SHOW_OUTPUT = true;

videowriter = VideoResultCreator('sar_recon_Video.avi', 20, 1, length(ALGORITHM_RUNLIST));
videowriter.setFontSize(16);

pol = 'HH';             % What polarization to image (HH,HV,VH,VV)

% Define image parameters here
data.Nfft = 4096;       % Number of samples in FFT
data.Nx = 512;          % Number of samples in x direction
data.Ny = 512;          % Number of samples in y direction
data.x0 = 0;            % Center of image scene in x direction (m)
data.y0 = 0;            % Center of image scene in y direction (m)
dyn_range = 70;         % dB of dynamic range to display

data.c = 299792458;     % Speed of light constant

if (strcmp(data_source,'GOTCHA') == true)
    dvdPath = '/home/arwillis/sar/GOTCHA/Gotcha-CP-All';
    
    % Define input data parameters here
    pass = 2;               % What pass to image (1-8)
    
    minaz = 20;             % Minimum azimuth angle (degrees)
    maxaz = 20;             % Maximum azimuth angle (degrees)
    af_flag = 0;            % Use autofocus flag (Only available for HH and VV)
    taper_flag = 0;         % Add a hamming taper for sidelobe control
    
    % Determine data path
    datadir = sprintf('%s%sDATA',dvdPath,filesep);
    
    % Read in the data
    for pulseIndex = minaz:maxaz
        % Determine file name based on input parameters
        in_fname = sprintf('%s%spass%d%s%s%sdata_3dsar_pass%d_az%03d_%s',datadir,...
            filesep,pass,filesep,pol,filesep,pass,pulseIndex,pol);
        
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
elseif (strcmp(data_source,'Sandia') == true)
    af_flag = 0;            % Use autofocus flag (Only available for HH and VV)
    taper_flag = 0;         % Add a hamming taper for sidelobe control
    
    addpath /home/arwillis/sar/Sandia/UUR_SPH_Utilities_v1.1/UUR_SPH_Utilities
    filename = '/home/arwillis/sar/Sandia/Rio_Grande_UUR_SAND2021-1834_O/SPH/PHX1T03_PS0008_PT000002.sph';
    sphObj=read_sph_stream(filename);
    
    totalRangeSamples = sphObj.preamble.Nsamples;
    totalPulses = double(sphObj.total_pulses);
    % 1 = HH, 2 = HV, 3 = VH, 4 = VVbandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
    switch (pol)
        case 'HH'
            channelIndex = 1;
        case 'HV'
            channelIndex = 2;
        case 'VH'
            channelIndex = 3;
        case 'VV'
            channelIndex = 4;
        otherwise
            warning('Unexpected polarization selected. Cannot select polarizations signal. Check the \"pol\" variable setting.')
    end
    numPulses = totalPulses;
    numRangeSamples = totalRangeSamples;
    numSamples = numRangeSamples;
    pulseIndices = 1+ceil((totalPulses - numPulses)/2):ceil((totalPulses + numPulses)/2);
    
    sphObj.goto_pulse(pulseIndices(1));
    sphObj.read_pulses(numPulses);
    data.phdata = sphObj.Data.SampleData(:,:,channelIndex);
    
    % allocate variables
    % variable to store the frequencies for each pulse
    data.vfreq = zeros(numSamples, numPulses);

    % variables to store the antenna (X,Y,Z) position for each pulse
    data.AntX = zeros(1, numPulses);
    data.AntY = zeros(1, numPulses);
    data.AntZ = zeros(1, numPulses);
    % variable to store the distance to target for each pulse
    data.R0 = zeros(1, numPulses);
    
    wgs84 = wgs84Ellipsoid('kilometer');
    antennaPos_ecef = zeros(numPulses,3);
    for pulseIndex = 1:numPulses
        velocity(pulseIndex,:) = [sphObj.Data.VelEast(pulseIndex), ...
            sphObj.Data.VelNorth(pulseIndex), ...
            sphObj.Data.VelDown(pulseIndex)];
            
        % antenna phase center offset from the radar
        antennaPos_geodetic = [sphObj.Data.RxPos.xat(pulseIndex), ...
            sphObj.Data.RxPos.yon(pulseIndex), ...
            sphObj.Data.RxPos.zae(pulseIndex)];
        
        % optical imagery = scene reference point G
        antennaPos(pulseIndex,:) = [sphObj.Data.radarCoordinateFrame.x(pulseIndex), ...
            sphObj.Data.radarCoordinateFrame.y(pulseIndex), ...
            sphObj.Data.radarCoordinateFrame.z(pulseIndex)];
        
        % Jetson Xavier platform
        % CUDA code cross-compile for Xavier
        % JSON for metadata
        % function array of positions, 
        
        rangeToTarget(pulseIndex) = norm(antennaPos(pulseIndex,:));
        startF = sphObj.Data.StartF(pulseIndex);
        freq_per_sec = sphObj.Data.ChirpRate(pulseIndex);
        freq_pre_sec_sq = sphObj.Data.ChirpRateDelta(pulseIndex);
        % Earth-Centered Earth-Fixed (ECEF)
        [antX, antY, antZ] = geodetic2ecef(wgs84, antennaPos_geodetic(1), ...
            antennaPos_geodetic(2), antennaPos_geodetic(3));
        antennaPos_ecef(pulseIndex,:) = [antX, antY, antZ];
        freq_per_sample(pulseIndex) = freq_per_sec/sphObj.preamble.ADF; % freq_per_sample
        pulseBandwidth(pulseIndex,:) = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
        bandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
        freq_samples = startF + bandwidth;  
        data.vfreq(:,pulseIndex) = freq_samples;
        data.R0(pulseIndex) = norm(antennaPos(pulseIndex,:));
        data.AntX(pulseIndex) = sphObj.Data.radarCoordinateFrame.x(pulseIndex);
        data.AntY(pulseIndex) = sphObj.Data.radarCoordinateFrame.x(pulseIndex);
        data.AntZ(pulseIndex) = sphObj.Data.radarCoordinateFrame.x(pulseIndex);
        
        minDistanceToTarget(pulseIndex) = sphObj.Data.RxTimeDelta(pulseIndex)*data.c;
        minDistanceToTarget2(pulseIndex) = sphObj.Data.TRDelay(pulseIndex)*data.c;
    end
    
    chirpRateDelta = sphObj.Data.ChirpRateDelta(:, channelIndex);
    startF = sphObj.Data.StartF(:, channelIndex);

    freq_per_sec = sphObj.Data.ChirpRate(pulseIndex);
    freq_pre_sec_sq = sphObj.Data.ChirpRateDelta(pulseIndex);
    
    data.AntX = sphObj.Data.radarCoordinateFrame.x(pulseIndices);
    data.AntY = sphObj.Data.radarCoordinateFrame.y(pulseIndices);
    data.AntZ = sphObj.Data.radarCoordinateFrame.z(pulseIndices);
    %data.AntX = antennaPos_ecef(:,1)';
    %data.AntY = antennaPos_ecef(:,2)';
    %data.AntZ = antennaPos_ecef(:,3)';
    %data.freq = startF + bandwidth;  
    data.freq = data.vfreq(:,1);

    analogToDigitalConverterFrequency = sphObj.preamble.ADF; % Hertz
    Ts0 = 1.0/analogToDigitalConverterFrequency;
    chirpRates_rad = sphObj.Data.ChirpRate(1, pulseIndices, channelIndex)*pi/180;
    nominalChirpRate = mean(chirpRates_rad);
    centerFreq_rad = sphObj.preamble.DesCntrFreq*pi/180;
    nominalChirpRate_rad = nominalChirpRate*pi/180;
    
    % use sphObj.Data.HwTimeCountLower for global aperture timing in SPH
    % files
    
end
%pulseIndices = (1:10)';
%[alphaX, alphaX_CI, resX, rintX, statsX] = regress(data.AntX(pulseIndices)', [ones(numel(pulseIndices),1) pulseIndices pulseIndices.^2])
%[alphaY, alphaY_CI, resY, rintY, statsY] = regress(data.AntY(pulseIndices)', [ones(numel(pulseIndices),1) pulseIndices pulseIndices.^2])
%[alphaZ, alphaZ_CI, resZ, rintZ, statsZ] = regress(data.AntZ(pulseIndices)', [ones(numel(pulseIndices),1) pulseIndices pulseIndices.^2])

% INPUT PARAMETERS END HERE %

%ph=uint8((255)*(((abs(data.phdata)./...
%            max(max(abs(data.phdata)))))))
%imshow(ph);
            
% Calculate the minimum frequency for each pulse (Hz)
data.minF = min(data.freq)*ones(size(data.R0));

% Calculate the frequency step size (Hz)
data.deltaF = diff(data.freq(1:2));

if af_flag
    % r_correct is a correction applied to r0 (effectively producing a
    % phase ramp for each pulse)
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
data.y_vec = linspace(data.y0 - data.Wy/2, data.y0 + data.Wy/2, data.Ny);
[data.x_mat,data.y_mat] = meshgrid(data.x_vec,data.y_vec);
data.z_mat = zeros(size(data.x_mat));

% Define speed of light (m/s)
c = 299792458;

% Determine the size of the phase history data
data.K = size(data.phdata,1);  % The number of frequency bins per pulse
data.Np = size(data.phdata,2); % The number of pulses

%figure(), plot(data.AntX, data.AntY);
%hold on;
trajectory = [data.AntX; data.AntY; data.AntZ];
centroid_xyz = mean(trajectory,2);
%theta=-atan2(data.AntY(1)-centroid_xyz(2), data.AntX(1)-centroid_xyz(1));
theta = -atan2(data.AntY(1), data.AntX(1));
%theta = theta*pi/180;
%theta = 0;
rotZ=[cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
newTrajectory = rotZ*trajectory;% - rotZ*centroid_xyz + centroid_xyz;
data.AntX = newTrajectory(1,:);
data.AntY = newTrajectory(2,:);
data.AntZ = newTrajectory(3,:);
%plot(data.AntX, data.AntY);

% Determine the azimuth angles of the image pulses (radians)
data.AntAz = unwrap(atan2(data.AntY,data.AntX));
% Determine the average azimuth angle step size (radians)
data.deltaAz = abs(mean(diff(data.AntAz)));

% Determine the total azimuth angle of the aperture (radians)
data.totalAz = max(data.AntAz) - min(data.AntAz);

% Determine the maximum scene size of the image (m)
data.maxWr = c/(2*data.deltaF);
data.maxWx = c/(2*data.deltaAz*mean(data.minF));

% Determine the resolution of the image (m)
data.dr = c/(2*data.deltaF*data.K);
data.dx = c/(2*data.totalAz*mean(data.minF));

% Display maximum scene size and resolution
fprintf('Maximum Scene Size:  %.2f m range, %.2f m cross-range\n',data.maxWr,data.maxWx);
fprintf('Resolution:  %.2fm range, %.2f m cross-range\n',data.dr,data.dx);

figureHandles(1) = figure(1);    

data.Np = min(200,data.Np);

Ishow = zeros(data.Ny,data.Nx,'uint8');

for runIndex=1:length(ALGORITHM_RUNLIST)
    ALGORITHM_RUNLIST(runIndex).time_profile = zeros(1,data.Np);
    ALGORITHM_RUNLIST(runIndex).entropy = zeros(1, data.Np);
    ALGORITHM_RUNLIST(runIndex).im_final = zeros(data.Ny, data.Nx);
end

% Loop through every pulse
for pulseIndex = 1:1:data.Np
    %figureHandles(runIndex) = figure(runIndex);
    % Display status of the imaging process
    if pulseIndex > 1 && mod(pulseIndex,50)==0
        t_sofar = 0;
        for runIndex=1:length(ALGORITHM_RUNLIST)
            t_sofar = t_sofar + sum(ALGORITHM_RUNLIST(runIndex).time_profile(1:(pulseIndex-1)));
        end
        t_est = (t_sofar*data.Np/(pulseIndex-1)-t_sofar)/60;
        fprintf('Pulse %d of %d, %.02f minutes remaining\n',pulseIndex,data.Np,t_est);
        dispstr = sprintf('Writing video to disk... frame %d',pulseIndex);
        disp(dispstr);
    end
    
    % This code runs if the frequency content of each pulse is different
    if (exist('data','var') == true && isfield(data,'vfreq') == true && ...
            size(data.vfreq,1) == data.K && size(data.vfreq,2) >= data.Np)
        % Set the frequencies that are in this pulse
        data.freq = data.vfreq(:, pulseIndex);
        data.freq_scales = 1 + (chirpRates_rad(pulseIndex)*Ts0/centerFreq_rad)*linspace(-data.K/2,data.K/2-1,data.K)';
        % Calculate the minimum frequency for each pulse (Hz)
        data.minF = min(data.vfreq);

        % Calculate the frequency step size (Hz)
        data.deltaF = diff(data.vfreq(1:2, pulseIndex));        
    end
        
    for runIndex=1:length(ALGORITHM_RUNLIST)
        
        if (max(ALGORITHM_RUNLIST(runIndex).pct_noise) > 0)
            data.AntX_true = data.AntX;
            data.AntY_true = data.AntY;
            data.AntZ_true = data.AntZ;
            pct_noise = ALGORITHM_RUNLIST(runIndex).pct_noise;
            if (isscalar(pct_noise))
                pct_noise = ones(1,3)*pct_noise;
            end
            data.lambda = data.c ./ data.freq;
            max_deviation = mean(data.lambda,'all')*pct_noise(1)/100;
            ALGORITHM_RUNLIST(runIndex).name = sprintf('%s U(+-%0.3g)cm noise', algorithm(2).name, 100*max_deviation);

            data.AntX = data.AntX + (pct_noise(1)/100)*(rand(size(data.AntX))-0.5)*2*mean(data.lambda,'all');
            data.AntY = data.AntY + (pct_noise(2)/100)*(rand(size(data.AntX))-0.5)*2*mean(data.lambda,'all');
            data.AntZ = data.AntZ + (pct_noise(3)/100)*(rand(size(data.AntX))-0.5)*2*mean(data.lambda,'all');
        end
        
        data.im_final = ALGORITHM_RUNLIST(runIndex).im_final;
        
        tic
        % Call the SAR focusing algorithm with the appropriate inputs
        pulse_contrib = ALGORITHM_RUNLIST(runIndex).algorithm_func(data, pulseIndex);
        % Determine the execution time for this pulse
        ALGORITHM_RUNLIST(runIndex).time_profile(pulseIndex) = toc;
        
        if (max(ALGORITHM_RUNLIST(runIndex).pct_noise) > 0)
            data.AntX = data.AntX_true;
            data.AntY = data.AntY_true;
            data.AntZ = data.AntZ_true;
        end
        
        % add the pulse contribution to the formed image
        ALGORITHM_RUNLIST(runIndex).im_final = ALGORITHM_RUNLIST(runIndex).im_final + pulse_contrib;   
        
        % process the output image for visualization
        im_final = ALGORITHM_RUNLIST(runIndex).im_final;
        Iout = (255/dyn_range)*((20*log10(abs(im_final)./...
            max(max(abs(im_final))))) + dyn_range);        
        title_str = sprintf("%s pulse=%d t(sec) = %0.1f",ALGORITHM_RUNLIST(runIndex).name, ...
            pulseIndex,sum(ALGORITHM_RUNLIST(runIndex).time_profile(1:(pulseIndex-1))));
        videowriter.setTitleString(title_str, 1, runIndex);
        Ishow = videowriter.addImage(uint8(Iout), 1, runIndex);

        ALGORITHM_RUNLIST(runIndex).Hcols(pulseIndex) = 0;
        for col = 1:1:size(Iout,2)
            ALGORITHM_RUNLIST(runIndex).Hcols(pulseIndex) = ALGORITHM_RUNLIST(runIndex).Hcols(pulseIndex) - sum(entropy(Iout(:,col)));
        end        
        
        %ALGORITHM_RUNLIST(runIndex).entropy(pulseIndex) = entropy(Iout);
        
        if (SHOW_OUTPUT == true && isempty(Ishow) == false)
            %Ishow = uint8(Iout);
            set(0, 'CurrentFigure', figureHandles(1));
            %set(groot,'CurrentFigure',figureHandles(runIndex));
            imshow(Ishow);
            %image('CData',Ishow);
            drawnow;
        end
        
    end
end
videowriter.close();

figure(length(figureHandles)+1)
for runIndex=1:length(ALGORITHM_RUNLIST)
   plot(ALGORITHM_RUNLIST(runIndex).Hcols);
   hold on;
end
title('Focused Image Entropy');
xlabel('Radar Pulse Index');
ylabel('H(I) - Entropy of focused image columns');
legend(ALGORITHM_RUNLIST(:).name)

data.im_final = ALGORITHM_RUNLIST(1).im_final;

% Display the image
figure(length(figureHandles)+2)
imagesc(data.x_vec,data.y_vec,20*log10(abs(data.im_final)./...
    max(max(abs(data.im_final)))),[-dyn_range 0])
colormap gray
axis xy image;
set(gca,'XTick',-data.Wx/2:round(data.Wx/10):data.Wx/2,'YTick',-data.Wy:round(data.Wy/10):data.Wy);
h = xlabel('x (m)');
set(h,'FontSize',14,'FontWeight','Bold');
h = ylabel('y (m)');
set(h,'FontSize',14,'FontWeight','Bold');
colorbar
set(gca,'FontSize',14,'FontWeight','Bold');
minaz = min(data.AntAz)*180/pi;
maxaz = max(data.AntAz)*180/pi;
%print -deps2 /ssip2/lgorham/SPIE10/fig/3DsarBPA.eps
out_fname = sprintf('3DsarBPA_az%0.0f-%0.0f_%s.png',minaz,maxaz,pol);
print(out_fname,'-dpng');
%clear data;


