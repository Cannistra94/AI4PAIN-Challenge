function [EDA_filt] = EDA_preproc(EDA,fs_EDA)

ordr     = 5;
ft       = 1; %[Hz] Cutoff frequency
Wn       = ft/(fs_EDA/2); %Normalized cutoff frequency
[b, a]   = butter(ordr,Wn);
EDA_filt = filtfilt(b,a,EDA);


end
