clear;
clc;

% Change these file paths to match your path setup
% dataFolder_reference = fullfile('Data');
%gt_data = csvread('gt_dataset_homography.csv');
execute_binary='../build/bin/autofocus_gpu';

% imds_reference = imageDatastore(dataFolder_reference, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds_reference = dir("Data");
imds_reference = imds_reference(3:end);

arg_reference_image='-i';
image_ref='';

estCoeff_list = [];
estTrajx_list = [];
estTrajy_list = [];
estTrajz_list = [];
gtx_list = [];
gty_list = [];
gtz_list = [];
runtime_list = [];
for fileIdx=1:length(imds_reference)
    time_start = tic;
    image_ref=imds_reference(fileIdx).name;
    command=strcat(execute_binary," ", ...
        arg_reference_image," ",strcat("Data/",image_ref))
    gt = load(strcat("Data/", image_ref));
    gtx_list = [gtx_list; gt.data.x(1:117)];
    gty_list = [gty_list; gt.data.y(1:117)];
    gtz_list = [gtz_list; gt.data.z(1:117)];
    [status,cmdout] = system(command);
    % add output of homography to cuGridSearch
    runtime = toc(time_start);
    result = regexp(cmdout,"MinParams\[([^\]]*)]", "tokens");
    estCoeff = reshape(str2num(result{1}{1}),1,[]);
    estTrajx = cumsum([gt.data.x(1), estCoeff(1) * (1:116) + estCoeff(2)]);
    estTrajy = cumsum([gt.data.y(1), estCoeff(3) * (1:116) + estCoeff(4)]);
    estTrajz = cumsum([gt.data.z(1), estCoeff(5) * (1:116) + estCoeff(6)]);
    estCoeff_list = [estCoeff_list; estCoeff];
    estTrajx_list = [estTrajx_list; estTrajx];
    estTrajy_list = [estTrajy_list; estTrajy];
    estTrajz_list = [estTrajz_list; estTrajz];
    runtime_list = [runtime_list; runtime];
end

% new_gtH = [];
% for fileIdx=1:length(imds_moving.Files)
%     gt_H = reshape(gt_data(fileIdx, 2:10), 3,3)';
%     scale = scale_list(fileIdx, :);
%     S = [scale(1) 0 0; 0 scale(2) 0; 0 0 1];
%     % Formula: B' = (S * H * S^-1) * S * A
%     resized_gtH = S * gt_H;
%     temp_gtH = reshape(resized_gtH', 1,[]);
%     new_gtH = [new_gtH; temp_gtH];
% end