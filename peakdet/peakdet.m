function [f0_times, f0_frames] = peakdet(file_path)

% setting resampling coefficient
resampC = 100;

% initializing matrix; assumption: there will be no more than 100 periods in each
% analyzed token; this value, which is sufficient for single syllables, 
% can be changed below, in order to treat longer intervals of voicing at one go:
MaxPerN = 100;
data(MaxPerN,10,1) = 0; 

% setting coefficient for recognition of "double peaks": for closings, is 0.5,
% following Henrich N., d'Alessandro C., Castellengo M. et Doval B., 2004, "On
% the use of the derivative of electroglottographic signals for characterization 
% of non-pathological voice phonation", Journal of the Acoustical Society of America, 
% 115(3), pp. 1321-1332.
propthresh = 0.5;


method = 1;

maxF = 500;

smoothingstep = 3;

% indicating path to EGG file
% disp('Please paste complete path to EGG file here (e.g. D:\EGGsession1\sig.wav)')
% pathEGG = input(' > ','s');
% [EGGfilename,pathEGG] = uigetfile('*.*','Please choose the EGG file to be downloaded');

% finding out the characteristics of the sound file

[Y,FS] = audioread(file_path);


COEF = [FS smoothingstep 1 0];

% setting the value for threshold for peak detection: by default, half
% the size of the maximum peak
propthresh = 0.5;



%%%%%%%%%%%%%% running main analysis programme
[Fo,Oq,Oqval,DEOPA,goodperiods,OqS,OqvalS,DEOPAS,goodperiodsS,simppeak,SIG,dSIG,SdSIG] = FO(COEF,Y,FS,method,propthresh,resampC,maxF);	


period = diff(simppeak(:, 1));

time = simppeak(2:end, 1);


period_sig = zeros(1, length(Y));

idx = round(time*FS);

period_sig(idx) = period;

window_size = 480;

half_window = window_size / 2;

period_sig = [zeros(1, half_window), period_sig, zeros(1, half_window)];

num_frames = floor(length(period_sig) / 160);

f0_frames = zeros(1, num_frames);

for i = 1: num_frames
    center_idx = (i - 1) * 160 + half_window + 1;
    stard_idx = center_idx - half_window;
    end_idx = center_idx + half_window;

    end_idx = min(end_idx, length(period_sig));
    period_seg = period_sig(stard_idx: end_idx);
    period_seg = period_seg(period_seg ~= 0);
    if length(period_seg) > 1
        mean_period = mean(period_seg);
        f0_frames(i) = 1 / mean_period;
    end

end

f0_times = 0:num_frames - 1;
f0_times = f0_times / 100;
end