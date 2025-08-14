    PPG_features_HR(pulse_basic,Ts_PPG,t_basic)

%% PPG


% Basic pulses
% HR and HRV, some basic morphological features
    if length(pulse_basic) > 5 % at least 10 heart beats are needed

        IBI_raw = (pulse_basic(:,2) - pulse_basic(:,1)) * Ts_PPG;
        t_IBI_raw = t_basic;

        % Method to remove outliers from Banhalmi et al. 2018
        % "Analysis of pulse rate variability measurement using a
        % smartphone camera"
        t_IBI = t_IBI_raw;
        MedIBI = median(IBI_raw);
        outliers = find(IBI_raw < MedIBI./1.2 | IBI_raw > MedIBI*1.2);
        t_IBI(outliers) = [];
        IBI = IBI_raw;
        IBI(outliers) = [];

        HR  = 60 ./ IBI;

        meanHR = mean(HR); %[bpm]

        SDNN = std(IBI) *1000; %[ms]

        diff_IBI = (IBI(2:end) - IBI(1:end-1)) *1000;

        RMSSD = rms(diff_IBI); %[ms]

        SDSD = std(diff_IBI);

        NN50 = length(find(abs(diff_IBI)>50)); % [ ]

        pNN50 = NN50 / length(IBI) * 100; % [%]

        NN20 = length(find(abs(diff_IBI)>20)); % [ ]

        pNN20 = NN20 / length(IBI) * 100; % [%]

        [SD1, SD2] = PoincarePlot(IBI); %[ms]

        SD1SD2 = SD1/SD2; %[ ]

        IBI_ms = IBI * 1000; %[ms]
        % Interpolation
        F_resample = 4;
        T_resample = 1/F_resample;
        t_res = t_IBI(1):T_resample:t_IBI(end);
        try
            IBI_ist_resample = interp1(t_IBI,IBI_ms,t_res,'spline');
            IBI_ist_demean = IBI_ist_resample - mean(IBI_ist_resample); % Remove mean
            % High pass filter
            ordr     = 6;
            ft       = 0.02; %[Hz] Cutoff frequency
            Wn_HP    = ft/(F_resample/2); %Normalized cutoff frequency
            [b, a]   = butter(ordr,Wn_HP,'high');
            IBI_HP   = filtfilt(b,a,IBI_ist_demean); % At least 18 points are needed to use filtfilt (time window = 4 sec)
            % Low pass filter
            ordr     = 6;
            ft       = 0.4; %[Hz] Cutoff frequency
            Wn_LP       = ft/(F_resample/2); %Normalized cutoff frequency
            [b, a]   = butter(ordr,Wn_LP);
            IBI_filt = filtfilt(b,a,IBI_HP);
            % From Elgendi - PPG signal analysis book
            NFFT = max(256,2^nextpow2(length(IBI_filt))); % the number of FFT
            [PSD,F] = pwelch(IBI_filt,length(t_res),length(t_res)/2,(NFFT*2)-1,F_resample);

            LF = [0.04 0.15]; %[ms^2]
            HF = [0.15 0.4]; %[ms^2]

            iLF = (round(F,2) >= LF(1)) & (round(F,2) <= LF(2));
            aLF  = trapz(F(iLF),PSD(iLF));

            iHF = (round(F,2) >= HF(1)) & (round(F,2) <= HF(2));
            aHF  = trapz(F(iHF),PSD(iHF));

            i_TP = (round(F,2) >= LF(1)) & (round(F,2) <= HF(2));
            TP   = trapz(F(i_TP),PSD(i_TP)); %[ms^2]

            LFHF_ratio = aLF/aHF; %[ms^2]

        catch

            aLF = 0;
            aHF = 0;
            TP = 0;
            LFHF_ratio = 0;

        end

        % Approximate Entropy
        dim  = 2; % Embedded dimension
        r1    = 0.2 * std(IBI_ms); % tolerance
        tau  = 1; % delay time
        ApEn = approxEntropy(dim, r1, IBI_ms, tau);


    else

        meanHR     = 0;
        SDNN       = 0;
        RMSSD      = 0;
        pNN50      = 0;
        SD1        = 0;
        SD2        = 0;
        SD1SD2     = 0;
        aLF        = 0;
        aHF        = 0;
        TP         = 0;
        LFHF_ratio = 0;
        ApEn       = 0;
        SDSD       = 0;
        NN50       = 0;
        NN20       = 0;
        pNN20      = 0;

    end

end
