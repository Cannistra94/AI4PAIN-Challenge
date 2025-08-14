clear all
close all
clc

main_dir = 'Dataset-20250414T153814Z-001\AI4Pain2025Dataset\train';

folders = dir(main_dir); 

folders = folders(3:end); 

bvp_dir = fullfile(main_dir,folders(1).name); 
bvp_folder = dir(bvp_dir);
bvp_folder = bvp_folder(3:end); 

eda_dir = fullfile(main_dir,folders(2).name); 
eda_folder = dir(eda_dir);
eda_folder = eda_folder(3:end); 

resp_dir = fullfile(main_dir,folders(3).name); 
resp_folder = dir(resp_dir); 
resp_folder = resp_folder(3:end); 

spo2_dir = fullfile(main_dir,folders(4).name); 
spo2_folder = dir(spo2_dir); 
spo2_folder = spo2_folder(3:end);

bvp = cell(1,length(bvp_folder)); 
eda = cell(1,length(eda_folder)); 
resp = cell(1,length(resp_folder)); 
spo2 = cell(1,length(spo2_folder)); 

for i = 1:length(bvp_folder)

    bvp{i} = readtable(fullfile(bvp_dir,bvp_folder(i).name));
    eda{i} = readtable(fullfile(eda_dir,eda_folder(i).name));
    resp{i} = readtable(fullfile(resp_dir,resp_folder(i).name));
    spo2{i} = readtable(fullfile(spo2_dir,spo2_folder(i).name));
   
end

%%

bvp_length = zeros(length(bvp),size(bvp{1},2)); 
eda_length = zeros(length(eda),size(eda{1},2)); 
resp_length = zeros(length(resp),size(resp{1},2));
spo2_length = zeros(length(spo2),size(spo2{1},2)); 

for i = 1:length(bvp)

    for j = 1:size(bvp{1},2)

        vector_bvp = table2array(bvp{i}(:,j)); 
        nan_values_bvp = sum(isnan(vector_bvp)); 
        bvp_length(i,j) = length(vector_bvp) - nan_values_bvp; 

        vector_eda = table2array(eda{i}(:,j)); 
        nan_values_eda = sum(isnan(vector_eda)); 
        eda_length(i,j) = length(vector_eda) - nan_values_eda; 

        vector_resp = table2array(resp{i}(:,j)); 
        nan_values_resp = sum(isnan(vector_resp)); 
        resp_length(i,j) = length(vector_resp) - nan_values_resp; 

        vector_spo2 = table2array(spo2{i}(:,j)); 
        nan_values_spo2 = sum(isnan(vector_spo2)); 
        spo2_length(i,j) = length(vector_spo2) - nan_values_spo2; 

    end

end

%%

colNames = {'Baseline','HIGH_1','REST_1','HIGH_2','REST_2','LOW_1','REST_3','LOW_2','REST_4','HIGH_3','REST_5','LOW_3','REST_6','LOW_4','REST_7','HIGH_4','REST_8','HIGH_5','REST_9','LOW_5','REST_10','LOW_6','REST_11','HIGH_6','REST_12','HIGH_7','REST_13','HIGH_8','REST_14','LOW_7','REST_15','LOW_8','REST_16','HIGH_9','REST_17','LOW_9','REST_18','LOW_10','REST_19','HIGH_10','REST_20','HIGH_11','REST_21','LOW_11','REST_22','LOW_12','REST_23','HIGH_12'};

table_bvp = array2table([zeros(1,48)], 'VariableNames',colNames); 
table_eda = array2table([zeros(1,48)], 'VariableNames',colNames); 
table_resp = array2table([zeros(1,48)], 'VariableNames',colNames); 
table_spo2 = array2table([zeros(1,48)], 'VariableNames',colNames); 

sub = [];

for i = 1:length(bvp)

    tbl_bvp = bvp{i}; 
    tbl_eda = eda{i}; 
    tbl_resp = resp{i}; 
    tbl_spo2 = spo2{i}; 

    colNames_tbl = tbl_bvp.Properties.VariableNames; %HP: all the signals for the same participant present the same columns

    tbl_temp_bvp = array2table([zeros(height(tbl_bvp),48)], 'VariableNames',colNames);
    tbl_temp_eda = array2table([zeros(height(tbl_eda),48)], 'VariableNames',colNames);
    tbl_temp_resp = array2table([zeros(height(tbl_resp),48)], 'VariableNames',colNames);
    tbl_temp_spo2 = array2table([zeros(height(tbl_spo2),48)], 'VariableNames',colNames);

    for j = 1:length(colNames)

        matchIdx = contains(colNames_tbl, colNames{j}); 
        matchIdx_find = find(matchIdx==1,1);
        tbl_temp_bvp(:,j) = tbl_bvp(:,matchIdx_find);
        tbl_temp_eda(:,j) = tbl_eda(:,matchIdx_find); 
        tbl_temp_resp(:,j) = tbl_resp(:,matchIdx_find); 
        tbl_temp_spo2(:,j) = tbl_spo2(:,matchIdx_find);

    end

    sub = [sub; i*ones(height(tbl_temp_bvp),1)]; 
    table_bvp = [table_bvp; tbl_temp_bvp];
    table_eda = [table_eda; tbl_temp_eda];
    table_resp = [table_resp; tbl_temp_resp];
    table_spo2 = [table_spo2; tbl_temp_spo2];

end

table_bvp(1,:) = []; 
table_eda(1,:) = []; 
table_resp(1,:) = []; 
table_spo2(1,:) = []; 

table_bvp = addvars(table_bvp,sub,'NewVariableNames','Subject','Before','Baseline');
table_eda = addvars(table_eda,sub,'NewVariableNames','Subject','Before','Baseline');
table_resp = addvars(table_resp,sub,'NewVariableNames','Subject','Before','Baseline');
table_spo2 = addvars(table_spo2,sub,'NewVariableNames','Subject','Before','Baseline');

dataMatrix_eda = table2array(table_eda(:,2:end)); 
figure
boxplot(dataMatrix_eda,'Labels',table_eda(:,2:end).Properties.VariableNames)

dataMatrix_spo2 = table2array(table_spo2(:,2:end)); 
figure
boxplot(dataMatrix_spo2,'Labels',table_spo2(:,2:end).Properties.VariableNames)

%% Signal processing
colNames_process = {'Baseline','HIGH_1','REST_1','HIGH_2','REST_2','LOW_1','REST_3','LOW_2','REST_4','HIGH_3','REST_5','LOW_3','REST_6','LOW_4','REST_7','HIGH_4','REST_8','HIGH_5','REST_9','LOW_5','REST_10','LOW_6','REST_11','HIGH_6','REST_12','HIGH_7','REST_13','HIGH_8','REST_14','LOW_7','REST_15','LOW_8','REST_16','HIGH_9','REST_17','LOW_9','REST_18','LOW_10','REST_19','HIGH_10','REST_20','HIGH_11','REST_21','LOW_11','REST_22','LOW_12','REST_23','HIGH_12'};

%% BVP - Preprocessing and feature extraction

colNames_bvp = {'meanHR','SDNN','RMSSD','SDSD','NN50','pNN50','NN20','pNN20','SD1','SD2','SD1SD2','LF','HF','TP','LFHF','ApEn'};

fs = 100;
Ts_PPG = 1/fs;

for i = 1:length(bvp)

    for j = 1:length(colNames_process)
    
    colNames_tbl = bvp{i}.Properties.VariableNames; %HP: all the signals for the same participant present the same columns
    matchIdx = contains(colNames_tbl, colNames_process{j});
    matchIdx_find = find(matchIdx==1,1);
        
    PPG = table2array(bvp{i}(:,matchIdx_find));
    PPG(isnan(PPG)) = [];

    PPG_filt = PPG_preproc(PPG,fs); 
    [sys_peak, I_peak, sys_foot, I_foot] = ab_detection(PPG_filt,Ts_PPG);

    I_foot_col = I_foot';

    pulse_basic = [I_foot_col(1:end-1) I_foot_col(2:end)];
    IBI_control_basic = (pulse_basic(:,2) - pulse_basic(:,1))*Ts_PPG;
    pulse_basic(IBI_control_basic<0.5 | IBI_control_basic>1.5,:) = [];

    t_total = (0:1:length(PPG_filt)-1) * Ts_PPG; 
    t_basic = t_total(pulse_basic(:,1));
    
    [meanHR(i,j), SDNN(i,j), RMSSD(i,j),SDSD(i,j),NN50(i,j),pNN50(i,j),NN20(i,j),pNN20(i,j),SD1(i,j),SD2(i,j),SD1SD2(i,j),aLF(i,j),aHF(i,j),TP(i,j),LFHFratio(i,j),ApEn(i,j)] = ...
            PPG_features_HR(pulse_basic,Ts_PPG,t_basic); 

    end

end

%% EDA - Preprocessing and feature extraction
colNames_eda = {'Fmax','SIE','meanEDA','stdEDA','slopeEDA','meanEDL','stdEDL','slopeEDL','meanampEDR','stdampEDR','freqEDR'};

% Normalize EDA on a subject level
subjects = unique(table_eda.Subject); 

mean_EDA = zeros(length(subjects),1);
std_EDA = zeros(length(subjects),1);

for i = 1:length(subjects)

    select_sub = find(table_eda.Subject==subjects(i));
    mat_sub = table2array(table_eda(select_sub,2:end)); 
    vec_sub = mat_sub(:);
    vec_sub(isnan(vec_sub)) = [];
    mean_EDA(i) = mean(vec_sub); 
    std_EDA(i) = std(vec_sub); 

end

for i = 1:length(eda)

    for j = 1:length(colNames_process)
    
    colNames_tbl = eda{i}.Properties.VariableNames; %HP: all the signals for the same participant present the same columns
    matchIdx = contains(colNames_tbl, colNames_process{j});
    matchIdx_find = find(matchIdx==1,1);
        
    EDA = table2array(eda{i}(:,matchIdx_find));
    EDA(isnan(EDA)) = [];

    EDA_filt = EDA_preproc(EDA,fs); 
    EDA_zscored = (EDA_filt - mean_EDA(i)) ./ std_EDA(i);
    
    [Fmax(i,j), SIE(i,j), meanEDA(i,j), stdEDA(i,j), slopeEDA(i,j), meanEDL(i,j), stdEDL(i,j), slopeEDL(i,j), meanampEDR(i,j), stdampEDR(i,j), freqEDR(i,j)] = EDA_features(fs,EDA_filt);

    end

end
