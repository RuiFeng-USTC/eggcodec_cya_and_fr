clear, clc;
addpath(genpath(pwd))

FS = 16000;

file_path = 'data_and_result/reco_lar.wav';

[f0_times, f0_frames] = peakdet(file_path);

% 按列存为txt 间隔用空格
writematrix([f0_times', f0_frames'], 'data_and_result/pitch_from_reco_egg.txt','Delimiter',' ');
