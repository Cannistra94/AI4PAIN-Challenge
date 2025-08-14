function [PulseAmpl,A1,A2,A,T1,T2] = PPG_features_morf_Basic(start_pulse_basic,end_pulse_basic,pulse_basic,Ts_PPG,PPG_filt)
% Basic morphological features

if(~isempty(start_pulse_basic) || ~isempty(end_pulse_basic))

    if (start_pulse_basic ~= 0) && (start_pulse_basic ~= end_pulse_basic)
        % Initialize variables

        PulseAmpl_ist = zeros(end_pulse_basic - start_pulse_basic,1);
        A1_ist = zeros(end_pulse_basic - start_pulse_basic,1);
        A2_ist = zeros(end_pulse_basic - start_pulse_basic,1);
        A_ist  = zeros(end_pulse_basic - start_pulse_basic,1);
        T1_ist = zeros(end_pulse_basic - start_pulse_basic,1);
        T2_ist = zeros(end_pulse_basic - start_pulse_basic,1);

        for t = 1:(end_pulse_basic - start_pulse_basic)

            n = start_pulse_basic + t - 1;

            PPG_pulse_basic_raw = PPG_filt(pulse_basic(n,1):pulse_basic(n,2));
            PPP_pulse_basic = PPG_pulse_basic_raw - min(PPG_pulse_basic_raw);
            PPG_pulse_basic_zscored_raw = zscore(PPG_filt(pulse_basic(n,1):pulse_basic(n,2)));
            PPG_pulse_basic_zscored = PPG_pulse_basic_zscored_raw - min(PPG_pulse_basic_zscored_raw);

            [peaks,I_peaks] = findpeaks(PPG_pulse_basic_zscored,'SortStr','descend');

            if (~isempty(I_peaks))
                PulseAmpl_ist(t) = peaks(1) - PPG_pulse_basic_zscored(1);
                A1_ist(t) = abs(trapz(PPG_pulse_basic_zscored(1:I_peaks(1))));
                A2_ist(t) = abs(trapz(PPG_pulse_basic_zscored(I_peaks(1):end)));
                A_ist(t)  = abs(trapz(PPG_pulse_basic_zscored));
                T1_ist(t) = I_peaks(1) * Ts_PPG;
                T2_ist(t) = (length(PPG_pulse_basic_zscored) - I_peaks(1)) * Ts_PPG;

            else
                PulseAmpl_ist(t) = 0;
                A1_ist(t) = 0;
                A2_ist(t) = 0;
                A_ist(t) = 0;
                T1_ist(t) = 0;
                T2_ist(t) = 0;
            end

        end

        PulseAmpl = nanmean(PulseAmpl_ist);
        A1 = nanmean(A1_ist);
        A2 = nanmean(A2_ist);
        A  = nanmean(A_ist);
        T1 = nanmean(T1_ist);
        T2 = nanmean(T2_ist);

    else

        PulseAmpl = 0;
        A1 = 0;
        A2 = 0;
        A  = 0;
        T1 = 0;
        T2 = 0;

    end
else

    PulseAmpl = 0;
    A1 = 0;
    A2 = 0;
    A  = 0;
    T1 = 0;
    T2 = 0;
end
end
