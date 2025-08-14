% Use basic and high quality classifier
clear 
close all
clc

load('.\OptimizedClassifiers\Basic_classifier.mat');
load('.\OptimizedClassifiers\High_classifier.mat');
load('Lab.mat')

% Change the path according to the local path
% path
main_dir = '\Data\Data_E4';

folders_raw = dir(main_dir);
folders = folders_raw(3:end);

for s = 1:length(folders)

    subject_dir = fullfile(main_dir,folders(s).name);
    dayfolders_raw = dir(subject_dir);
    dayfolders = dayfolders_raw(3:end);

    for d = 1:length(dayfolders)
        
        clear ACC_g pp_amp mean_amp Sig_sim E Kur N_Elg RelP S Z ampl width trough_depth MedianSig_zscored MedianSig_nozscored MeanSig StdSig SNR  Npeaks Z_der FeaturesNames_basic Matrix_basic Table_basic FeaturesNames_high Matrix_high Table_high

        parent_dir = fullfile(subject_dir,dayfolders(d).name);

        PPGraw = load(fullfile(parent_dir,'BVP.csv'));
        ACCraw = load(fullfile(parent_dir,'ACC.csv'));

%         template_struct = load('Template_v3.m','-mat');
%         template = template_struct.template;

        start = datetime(PPGraw(1),'convertfrom','posixtime','timezone','Europe/Rome');

        PPG    = PPGraw(3:end);
        N_PPG  = length(PPG);
        fs_PPG = PPGraw(2);
        Ts_PPG = 1/fs_PPG;
        t_PPGs = (0:1:N_PPG - 1) * Ts_PPG;
        t_PPG  = start + seconds(t_PPGs);

        ACC    = ACCraw(3:end,:);
        N_ACC  = length(ACC);
        fs_ACC = ACCraw(2,1);
        Ts_ACC = 1/fs_ACC;
        t_ACCs = (0:1:N_ACC - 1) * Ts_ACC;
        t_ACC  = start + seconds(t_ACCs);

        g = 9.8; %[m/s^2]
        ACC_g(:,1) = ACC(:,1) / (64);
        ACC_g(:,2) = ACC(:,2) / (64);
        ACC_g(:,3) = ACC(:,3) / (64);

        % Modulus from the three accelerometer components
        ACC_mod = sqrt(ACC_g(:,1).^2 + ACC_g(:,2).^2 + ACC_g(:,3).^2);

        %%

        PPG_filt = PPG_preproc(PPG,fs_PPG);
        L        = length(PPG_filt);

        [sys_peak, I_peak, sys_foot, I_foot] = ab_detection(PPG_filt,Ts_PPG);

        I_foot_acc = round((I_foot .* Ts_PPG) .* fs_ACC);

        pp_amp = zeros(length(I_foot)-1,1);
        mean_amp = zeros(length(I_foot)-1,1);
        Sig_sim = zeros(length(I_foot)-1,1);
        E = zeros(length(I_foot)-1,1);
        N_Elg = zeros(length(I_foot)-1,1);
        RelP = zeros(length(I_foot)-1,1);
        S = zeros(length(I_foot)-1,1);
        Kur= zeros(length(I_foot)-1,1);
        Z = zeros(length(I_foot)-1,1);
        ampl= zeros(length(I_foot)-1,1);
        width = zeros(length(I_foot)-1,1);
        trough_depth = zeros(length(I_foot)-1,1);
        MedianSig_nozscored= zeros(length(I_foot)-1,1);
        MedianSig_zscored = zeros(length(I_foot)-1,1);
        MeanSig = zeros(length(I_foot)-1,1);
        StdSig = zeros(length(I_foot)-1,1);
        SNR = zeros(length(I_foot)-1,1);
        Npeaks = zeros(length(I_foot)-1,1);
        Z_der = zeros(length(I_foot)-1,1);

        pos_foot = zeros(length(I_foot)-1,2);
        
        for i = 1:length(I_foot) - 1

            pulse_analysis_nozscore = PPG_filt(I_foot(i):I_foot(i+1));

            pulse_analysis = zscore(PPG_filt(I_foot(i):I_foot(i+1)));
            P = length(pulse_analysis);

            pulse_analysis(pulse_analysis == 0) = 0.01;
            %In such this way, log(0) is replaced by log(0.01), that is not -Inf

            % Signal Quality

            % Accelerometer peak2peak amplitude
            try
                pp_amp(i) = peak2peak(ACC_mod(I_foot_acc(i):I_foot_acc(i+1)));
                mean_amp(i) = mean(ACC_mod(I_foot_acc(i):I_foot_acc(i+1)));
            catch
                pp_amp(i) = pp_amp(i-1);
                mean_amp(i) = mean_amp(i-1);
            end

            % By Jang18: Correlation coefficient between two adjancent pulses
            if i == 1
                Sig_sim(i) = 1;
            else
                pulse_analysis_prev = PPG_filt(I_foot(i-1):I_foot(i));
                if length(pulse_analysis_prev) < length(pulse_analysis)
                    pulse_analysis_prev_2 = interp1(1:numel(pulse_analysis_prev),pulse_analysis_prev,linspace(1,numel(pulse_analysis_prev),numel(pulse_analysis)));
                    Sig_sim_mat = corrcoef(pulse_analysis_prev_2,pulse_analysis);
                    Sig_sim(i) = Sig_sim_mat(1,2);
                elseif length(pulse_analysis_prev) > length(pulse_analysis)
                    pulse_analysis_2 = interp1(1:numel(pulse_analysis),pulse_analysis,linspace(1,numel(pulse_analysis),numel(pulse_analysis_prev)));
                    Sig_sim_mat = corrcoef(pulse_analysis_prev,pulse_analysis_2);
                    Sig_sim(i) = Sig_sim_mat(1,2);
                else
                    Sig_sim_mat = corrcoef(pulse_analysis_prev,pulse_analysis);
                    Sig_sim(i) = Sig_sim_mat(1,2);
                end
            end

            %Entropy
            E(i) = - sum(pulse_analysis.^2 .* log(pulse_analysis.^2));

            % Signal-to-noise ratio
            N_Elg(i) = std(abs(pulse_analysis)).^2 / std(pulse_analysis);

             % Relative Power
            [PSD,F] = pwelch(pulse_analysis,[],[],[],fs_PPG);

            f_1Hz = 1; %Hz
            f1    = 2.25; %Hz
            f2    = 8; %Hz

            index_f1Hz = find(F == f_1Hz);
            index_f1   = find(F == f1);
            index_f2   = find(F == f2);

            RelP(i) = sum(PSD(index_f1Hz:index_f1))/sum(PSD(1:index_f2));

            % Skewness
            mean_sd = mean(pulse_analysis) / std(pulse_analysis);
            vec_mean_sd = mean_sd * ones(size(pulse_analysis));

            S(i) = 1/P * sum((pulse_analysis - vec_mean_sd).^3)';

             % Kurtosis
            Kur(i) = 1/P * sum((pulse_analysis - vec_mean_sd).^4);

            % Zero Crossing rate
            signal_zerocross = length(find (pulse_analysis < 0));

            Z(i) = 1/P * sum(signal_zerocross);

            % By Sukor11: morphological features
            ampl(i) = abs(sys_peak(i) - sys_foot(i));

            width(i) = length(pulse_analysis);

            trough_depth(i) = abs(pulse_analysis(1) - pulse_analysis(end));

            % Statistical parameters
           
            MedianSig_nozscored(i) = median(pulse_analysis_nozscore);

            MedianSig_zscored(i) = median(pulse_analysis);

            MeanSig(i) = mean(pulse_analysis_nozscore);

            StdSig(i) = std(pulse_analysis_nozscore);

            % By Moody's algorithm (?)
            SNR(i) = moody(pulse_analysis);

            % Updated 18.05
            % # of detected peaks
            [pks] = findpeaks(pulse_analysis);
            if isempty(pks)
                Npeaks(i) = 0;
            else
                Npeaks(i) = length(pks);
            end

            % Derivative zerocrossing rate
            der = diff(pulse_analysis);
            P_der = length(der);
            der_zerocross = length(find (der < 0));

            Z_der(i) = 1/P_der * sum(der_zerocross);

            pos_foot(i,1) = I_foot(i);
            pos_foot(i,2) = I_foot(i+1);
        end

        FeaturesNames_basic = Lab;
        Matrix = [pp_amp, mean_amp, Sig_sim, E, Kur, N_Elg, RelP, S, Z, ampl, width, trough_depth, MedianSig_zscored, MedianSig_nozscored, MeanSig, StdSig, SNR,  Npeaks, Z_der];

        % Z-score
        Matrix_zscored = zeros(size(Matrix));

        for i = 1:size(Matrix,2)

            Matrix_zscored(:,i) = (Matrix(:,i) - mu(i)) ./ sigma(i);
                
        end

        Table_tobeclassified = array2table(Matrix_zscored,'VariableNames',FeaturesNames_basic);

        %% Basic classifier
        % y_basic = 1 if the pulse can be used to estimate the HR, 0 otherwise
        y_basic_double = basic_classifier.predictFcn(Table_tobeclassified);
        y_basic = logical(round(y_basic_double));

        % Find the foot indices of pulses that can be used to estimate the HR
        ind_basic = find(y_basic == 1);

        % E.g.: Interbeat Interval
        % IBI = (I_foot(ind_basic+1) - I_foot(ind_basic)) .* Ts_PPG;

        %% High classifier
        % y_high = 1 if the pulse can be used for morphological analysis, 0 otherwise
        y_high = high_classifier.predictFcn(Table_tobeclassified);

        ind_high = find(y_high == 2);

        % for i = 1:length(ind_high) - 1
        %
        %     % E.g.: Crest Time
        %     CT(i) = (I_peak(ind_high(i)+1) - I_foot(ind_high(i))) * Ts_PPG;
        %     % + 1 in I_peak because the ab_detection algorithm detect the I_peak
        %     % first and then the I_foot
        %
        % end

%         foot_and_label_basic = zeros(size(PPG,1),1);
%         foot_and_label_basic(I_foot,1) = 1;
%         foot_and_label_basic(I_foot(ind_basic),2) = 1;
%         foot_and_label_basic = find(foot_and_label_basic(:,1) == 1);
%         foot_and_label_high = zeros(size(PPG,1),1);
%         foot_and_label_high(I_foot,1) = 1;
%         foot_and_label_high(I_foot(ind_high),2) = 1;
%         foot_loc_high = find(foot_and_label_high(:,1) == 1);
%                 
%             figure
%             plot(PPG_filt)
%             hold on 
%             for i = 1:size(y_basic,1)
%                 if y_basic(i) == 1
%                     plot(I_foot(i):I_foot(i+1), PPG_filt(I_foot(i):I_foot(i+1)), 'r') %'LineWidth', 12)
%                 end
%             end
%             plot(I_foot, PPG_filt(I_foot), 'k*')
%             
%             figure
%             plot(PPG_filt)
%             hold on 
%             for i = 1:size(y_high,1)
%                 if y_high(i) == 1
%                     plot(I_foot(i):I_foot(i+1), PPG_filt(I_foot(i):I_foot(i+1)), 'r') %'LineWidth', 12)
%                 end
%             end
%             plot(I_foot, PPG_filt(I_foot), 'k*')
% 
%             pause

        %% Save
        save(fullfile(parent_dir,'ind_basic.mat'),'ind_basic');
        save(fullfile(parent_dir,'ind_high.mat'),'ind_high');
        save(fullfile(parent_dir,'I_foot.mat'),'I_foot');
    end
disp(['Done with subject ',num2str(s),' out of ',num2str(length(folders))])
end
