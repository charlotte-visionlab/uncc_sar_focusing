clear;
% clc;

% Change these file paths to match your path setup
% dataFolder_reference = fullfile('Data');
%gt_data = csvread('gt_dataset_homography.csv');

num_search_recursions = '1';
PLOT_TRAJECTORY = true;
DATASET_INDEX = 1;
USER_FOLDER = 'cbeam18';

% Beam - HOME SITE CONFIG
execute_binary='../build/bin/autofocus_gpu';
%dataFolder_root = fullfile('/home', USER_FOLDER, 'CLionProjects', 'sar_focusing_exps');
%dataFolder_results = fullfile('/home', USER_FOLDER,  'CLionProjects', 'sar_focusing_exps', 'results');
dataFolder_dataset = fullfile('Data');
dataFolder_results = fullfile('OutputImages');
library_path='../build/lib';

% Willis - UNIVERSITY SITE CONFIG
%dataFolder_root = fullfile('/home', USER_FOLDER, 'CLionProjects', 'georeg_exps');
%dataFolder_root = fullfile('/home','server', 'SAR');
% Willis - HOME SITE CONFIG
% dataFolder_dataset = fullfile('/home', USER_FOLDER, 'sar', 'GOTCHA', 'Gotcha-CP-All');
% dataFolder_results = fullfile('/home', USER_FOLDER,  'CLionProjects', 'sar_focusing_exps', 'results');
% execute_binary='../cmake-build-debug/bin/autofocus_gpu';
% library_path='../cmake-build-debug/lib';

extList{1} = '.mat';
imds = imageDatastore(dataFolder_dataset, 'IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions', extList);

models{1} = 'Linear';
models{2} = 'Quadratic';

f1_handle = figure('visible','off');

%for modelIdx=2:length(models)
%for modelIdx=1:1
for modelIdx=1:2
    
    if (strcmp(models{modelIdx},'Linear')==1)
        model_type = "0";
        nPulses = "30";
    else
        model_type = "1";
        nPulses = "117";
    end
    
    num_focused_pulses=str2num(nPulses);
    
    for fileIdx=1:length(imds.Files)
        image_ref=imds.Files{fileIdx};
        filename_path = regexp(image_ref,"(.*\/).*", "tokens");
        filename_source = image_ref(length(filename_path{1}{1})+1:end);
        basename_source = filename_source(1:end-4);
        output_name = sprintf("%s/%s_style_%s_numPulses_%s.bmp", ...
            dataFolder_results, basename_source, model_type, nPulses);

        command=strcat(execute_binary," ", ...
            "-i", " ", image_ref, " ", ...
            "-m", " ", num_search_recursions, " ", ...
            "-s", " ", model_type, " ", ...
            "-n", " ", nPulses, " ", ...
            "-o", " ", output_name)
        
        time_start = tic;
        command = strcat('export LD_LIBRARY_PATH=', library_path, ' ; ', command);
        [status, cmdout] = system(command);
        runtime = toc(time_start);
        
        grid_values_str = regexp(cmdout,"\{(.*?)\}", "tokens");
        grid_values.start = str2num(grid_values_str{26}{1});
        grid_values.stop = str2num(grid_values_str{27}{1});
        grid_values.stepsize = str2num(grid_values_str{28}{1});
        grid_values.numsteps = str2num(grid_values_str{29}{1});
        result = regexp(cmdout,"MinParams\[([^\]]*)]", "tokens");
        covarText = regexp(cmdout,"Covariance matrix\[([^\]]*)]", "tokens");
        estCoeff = reshape(str2num(result{1}{1}),1,[]);
        covarMat = reshape(str2num(covarText{1}{1}),1,[]);
        if (strcmp(models{modelIdx},'Linear')==1)
            estTrajx = estCoeff(2) * (1:num_focused_pulses) + estCoeff(1);
            estTrajy = estCoeff(4) * (1:num_focused_pulses) + estCoeff(3);
            estTrajz = estCoeff(6) * (1:num_focused_pulses) + estCoeff(5);
        else
            estTrajx = estCoeff(3) * (1:num_focused_pulses).^2 + estCoeff(2) * (1:num_focused_pulses) + estCoeff(1);
            estTrajy = estCoeff(6) * (1:num_focused_pulses).^2 + estCoeff(5) * (1:num_focused_pulses) + estCoeff(4);
            estTrajz = estCoeff(9) * (1:num_focused_pulses).^2 + estCoeff(8) * (1:num_focused_pulses) + estCoeff(7);
        end
        trajectory_3D_estimated = [estTrajx', estTrajy', estTrajz'];

        GOTCHA_phase_history = load(image_ref);
        trajectory_3D_ground_truth = [GOTCHA_phase_history.data.x(1:num_focused_pulses)', ...
            GOTCHA_phase_history.data.y(1:num_focused_pulses)', ...
            GOTCHA_phase_history.data.z(1:num_focused_pulses)'];
        if (PLOT_TRAJECTORY == true)
            set(0,'CurrentFigure',f1_handle);
            set(f1_handle, 'Visible', 'on');
            %trajectory_3D_estimated(1:10,:)
            %trajectory_3D_ground_truth(1:10,:)
            hold off;
            plot3(trajectory_3D_ground_truth(:,1), trajectory_3D_ground_truth(:,2), trajectory_3D_ground_truth(:,3),'b');
            hold on;
            plot3(trajectory_3D_estimated(:,1), trajectory_3D_estimated(:,2), trajectory_3D_estimated(:,3),'r');
            drawnow;
        end
        
        % Get covariance, look into orthogonal-ness of the lines (figure
        % out why it's not lining up), check final number (metric)
        
        dataset(modelIdx).results(fileIdx).filename = image_ref;
        dataset(modelIdx).results(fileIdx).grid = grid_values;
        dataset(modelIdx).results(fileIdx).model_coefficients = estCoeff;
        dataset(modelIdx).results(fileIdx).trajectory_3D_ground_truth = trajectory_3D_ground_truth;
        dataset(modelIdx).results(fileIdx).trajectory_3D_estimated = trajectory_3D_estimated;
        dataset(modelIdx).results(fileIdx).num_focused_pulses = num_focused_pulses;
        dataset(modelIdx).results(fileIdx).multiresolution = num_search_recursions;
        dataset(modelIdx).results(fileIdx).model_type = models{modelIdx};
        dataset(modelIdx).results(fileIdx).covariance_matrix = covarMat;
        
        dataset(modelIdx).match(fileIdx).runtime = runtime;
    end
end

dataset(1).date = date;
experiment_filename_str = sprintf('sar_focusing_exp_results-%s.mat', dataset(1).date)
save(experiment_filename_str,'dataset')

