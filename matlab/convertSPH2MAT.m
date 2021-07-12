clear;
clc;
addpath ../../Sandia/UUR_SPH_Utilities_v1.1/UUR_SPH_Utilities
fileprefix = '../../Sandia/Rio_Grande_UUR_SAND2021-1834_O/SPH/PHX1T03_PS0008_PT0000';
filepostfix = '';
%fileprefix = '../../Sandia/Farms_UUR_SAND2021-1835_O/SPH/0506P19_PS0020_PT0000';
%filepostfix = '_N03_M1';
for idx=1:1
    infilename = sprintf('%s%02d%s.sph',fileprefix,idx,filepostfix);
    outfilename = sprintf('%s%02d%s.mat',fileprefix,idx,filepostfix);
    sphObj=read_sph_stream(infilename);
    totalPulses = double(sphObj.total_pulses);
    sphObj.read_pulses(totalPulses);
    sph_MATData.filenames = sphObj.filenames;
    sph_MATData.total_pulses = sphObj.total_pulses;
    sph_MATData.preamble = sphObj.preamble;
    sph_MATData.Const = sphObj.Const;
    sph_MATData.Data = sphObj.Data;
    % if you do not compress, you get larger file sizes and a shorter load time
    save(outfilename,'sph_MATData','-v7','-nocompression');
    % if you compress, you get smaller file sizes but a longer load time
    % save(outfilename,'sph_MATData','-v7');
end
