import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import neurokit2 as nk
import math

import scipy.signal as sg
from scipy.fft import fft
from scipy.signal.windows import blackman
from scipy.stats import entropy

plt.style.use("seaborn-v0_8-whitegrid")
import seaborn as sns

mpl.rcParams["lines.linewidth"] = 0.91
sns.set_context("talk")

import sys
sys.path.append('AI4PAIN_challenge/Data_Preparation')

import ppg_processing
from ppg_processing import filter_ppg, MSPTDfast, get_peak_onset
from ppg_fiducials import *

# EDA functions
def EDA_features(eda, fs):
    t_eda = np.arange(len(eda)) / fs
    cleaned = nk.eda_clean(eda, sampling_rate=fs)
    eda_decomposed = nk.eda_phasic(cleaned, sampling_rate=fs, method="cvxeda")
    tonic = np.array(eda_decomposed["EDA_Tonic"])
    phasic = np.array(eda_decomposed["EDA_Phasic"])

    # Fmax (Quintero et al., 2017)
    hop = np.arange(0, len(cleaned), 200)
    psd_eda = []

    for j in range(len(hop) - 1):
        if hop[j] + 399 < len(cleaned):
            segment = cleaned[hop[j] : hop[j] + 400]
        else:
            segment = cleaned[hop[j] :]

        segment = np.asarray(segment, dtype=np.float64)
        window = blackman(len(segment))
        eda_wind = np.asarray(segment * window, dtype=np.float64)
        eda_spec = fft(eda_wind, n=len(eda))
        freq = np.arange(len(eda_spec)) * 4 / len(eda)
        power = (1 / 4) * np.abs(eda_spec[: len(eda_spec) // 2]) ** 2
        psd_eda.append(power)

    psd_eda = np.array(psd_eda)
    periodogram = np.mean(psd_eda, axis=0)
    freq_2 = freq[: len(freq) // 2]
    totPow = np.trapz(periodogram, freq_2)

    Fmax = None
    for k in range(len(freq_2)):
        Pow = np.trapz(periodogram[k:], freq_2[k:])
        if Pow / totPow < 0.05:
            Fmax = freq_2[k]
            break

    # Symbolic Information Entropy (SIE)
    diff_eda = np.diff(eda) / np.diff(t_eda)
    SD_diff = np.std(diff_eda)
    sym = np.zeros(len(eda) - 1)

    for p in range(len(sym)):
        delta = eda[p + 1] - eda[p]
        if delta >= 0 and delta < SD_diff:
            sym[p] = 0
        elif delta >= 0 and delta >= SD_diff:
            sym[p] = 1
        elif delta < 0 and delta >= -SD_diff:
            sym[p] = 2
        else:
            sym[p] = 3

    L = len(sym)
    m = 2
    X = np.array([sym[i : i + m] for i in range(L - m + 1)])
    X_merged = [int(f"{int(a)}{int(b)}") for a, b in X]

    mode = np.array([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
    prob_mode = np.array([X_merged.count(mo) / (L - m + 1) for mo in mode])
    P = np.log2(prob_mode, where=prob_mode > 0)
    SIE = -np.sum(prob_mode * P)

    # EDA total mean, std, slope
    meanEDA = np.mean(eda)
    stdEDA = np.std(eda)
    slopeEDA = np.polyfit(t_eda, eda, 1)[0]

    # EDL mean, std, slope
    meanEDL = np.mean(tonic)
    stdEDL = np.std(tonic)
    slopeEDL = np.polyfit(t_eda, tonic, 1)[0]

    # EDR metrics
    der_phasic = np.diff(phasic) / np.diff(t_eda)
    peaks, _ = sg.find_peaks(der_phasic, height=0.01, distance=fs)

    phasic_peaks = []
    for loc in peaks:
        local_peaks, _ = sg.find_peaks(phasic[loc:])
        if local_peaks.size > 0:
            phasic_peaks.append(phasic[loc + local_peaks[0]])

    if len(phasic_peaks) == 0:
        meanampEDR = 0
        stdampEDR = 0
        freqEDR = 0
    else:
        meanampEDR = np.mean(phasic_peaks)
        stdampEDR = np.std(phasic_peaks)
        freqEDR = len(phasic_peaks) / (t_eda[-1] / 60)

    features = {
        "EDA_Fmax": Fmax,
        "EDA_SIE": SIE,
        "EDA_Mean": meanEDA,
        "EDA_Std": stdEDA,
        "EDA_Slope": slopeEDA,
        "EDL_Mean": meanEDL,
        "EDL_Std": stdEDL,
        "EDL_Slope": slopeEDL,
        "EDR_MeanAmp": meanampEDR,
        "EDR_StdAmp": stdampEDR,
        "EDR_Freq": freqEDR,
    }
    return pd.DataFrame([features])


def statistic_feature(signal, fs):

    # Peak-to-RMS
    p2rms = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))

    # Num of peaks
    # peaks, _ = signal.find_peaks(signal)
    # num_peaks = len(peaks)

    # Shannon Entropy (approximate using normalized histogram)
    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist[hist > 0]  # avoid log(0)
    ShEn = -np.sum(hist * np.log2(hist))

    # Log energy entropy
    LogEn = np.sum(np.log(signal**2 + np.finfo(float).eps))

    # Mean of the first derivative (slope)
    t_signal = np.arange(len(signal)) / fs
    mean_slope = np.polyfit(t_signal, signal, 1)[0]

    # Root Mean Square (RMS)
    rms = np.sqrt(np.mean(signal**2))

    # peak-to-peak
    p2p = np.ptp(signal)

    features = {
        "Stat_ShannonEntropy": ShEn,
        "Stat_LogEnergyEntropy": LogEn,
        "Stat_Peak2RMS": p2rms,
        "Stat_MeanSlope": mean_slope,
        "Stat_RMS": rms,
        "Stat_PeakToPeak": p2p,
    }

    features = {k: float(v) for k, v in features.items()}

    return pd.DataFrame([features])

# PPG morphological and HRV features
import warnings
warnings.filterwarnings("ignore")

# MS's path
# ppg_path = "AI4Pain 2025 Dataset/train_validation/bvp/"


subjects = sorted(os.listdir(ppg_path))

fs = 100
all_dfs = []

features_df_morph = pd.DataFrame()
features_df_morph_normalized = pd.DataFrame()

for i, sub in enumerate(subjects):
    print(f"Processing subject {i+1}/{len(subjects)}: {sub}")
    ppg = pd.read_csv(os.path.join(ppg_path, sub), header=0)
    original_lengths = ppg.notna().sum().to_dict()

    # Concatenate all columns and drop NaN values
    ppg_full = np.concatenate([ppg[col].dropna().values for col in ppg.columns])
    ppg_full_filtered = filter_ppg(ppg_full)

    # Reshape the filtered data back to the original columns
    reshaped_data = {}
    start = 0
    for col in ppg.columns:
        length = original_lengths[col]
        segment = ppg_full_filtered[start : start + length]
        padded_segment = np.full(ppg.shape[0], np.nan)
        padded_segment[:length] = segment
        reshaped_data[col] = padded_segment
        start += length

    ppg_filtered_df = pd.DataFrame(reshaped_data)

    ppg_full_filtered = np.concatenate(
        [ppg_filtered_df[col].dropna().values for col in ppg_filtered_df.columns]
    )

    # Apply Aboy algorithm
    peaks, onsets = get_peak_onset(ppg_full_filtered, fs=100)

    # Loop through each trial and extract features
    trial_peaks, trial_onsets = {}, {}
    samples_to_keep = 10 * fs  # 10 seconds for rest trials
    index = 0
    for col in ppg_filtered_df.columns:
        signal = ppg_filtered_df[col].dropna().values
        length = len(signal)
        end = index + length

        # Default values
        sig_offset = 0
        signal_final = signal

        # If REST trial: keep only last 10 seconds
        if "REST" in col:
            if length > samples_to_keep:
                sig_offset = length - samples_to_keep
                signal_final = signal[sig_offset:]
            else:
                sig_offset = 0
                signal_final = signal  # keep all if too short

                # If REST trial: keep only last 10 seconds
        if "BASELINE" in col:
            if length > samples_to_keep:
                sig_offset = length - samples_to_keep
                signal_final = signal[sig_offset:]
            else:
                sig_offset = 0
                signal_final = signal  # keep all if too short

        # Adjust peaks and onsets to the shortened window
        peaks_in_range = [
            p - index - sig_offset for p in peaks if index + sig_offset <= p < end
        ]
        onsets_in_range = [
            o - index - sig_offset for o in onsets if index + sig_offset <= o < end
        ]

        trial_peaks[col] = peaks_in_range
        trial_onsets[col] = onsets_in_range

        index = end

    # Compute HR and PWA
    sampling_rate = 100
    features_dict = {}
    for col in ppg_filtered_df.columns:
        signal = ppg_filtered_df[col].dropna().values
        peaks = trial_peaks[col]
        onsets = trial_onsets[col]

        # HR computation
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / sampling_rate
            max_rr = np.max(rr_intervals)
            min_rr = np.min(rr_intervals)
            mean_rr = np.mean(rr_intervals)
            mean_hr = 60 / mean_rr
            # print(mean_hr, mean_rr)
        else:
            mean_rr = np.nan
            mean_hr = np.nan
        # PWA computation: peak - preceding onset
        pairs = [(o, p) for o in onsets for p in peaks if o < p]
        nearest_pairs = {}
        for o, p in pairs:
            if p not in nearest_pairs or (p - o) < (p - nearest_pairs[p]):
                nearest_pairs[p] = o
        amplitudes = [
            signal[p] - signal[o]
            for p, o in nearest_pairs.items()
            if 0 <= o < len(signal) and 0 <= p < len(signal)
        ]
        pwa = np.mean(amplitudes) if amplitudes else np.nan
        # I always need to start with an onset and end with another onset
        if (
            peaks[0] < onsets[0]
        ):  # If the first peak is before the first onset, we need to remove it
            peaks = peaks[1:]  # Remove the first peak
        if (
            peaks[-1] > onsets[-1]
        ):  # If the last peak is after the last onset, we need to remove it
            peaks = peaks[:-1]  # Remove the last peak

        # Morphological features (!!NOTE!! Might make sense to interpolate pulses to 100 samples to avoid redundancy with HR features)
        dn = get_dicrotic_notch(signal, fs=100, peaks=np.array(peaks), onsets=onsets)
        apg_fp = get_apg_fiducials(signal, peaks=np.array(peaks), onsets=onsets)
        e = apg_fp["e"].values.astype(int)
        a = apg_fp["a"].values.astype(int)
        b = apg_fp["b"].values.astype(int)

        dp = get_diastolic_peak(signal, onsets, dn, pd.Series(e)).astype(int)
        dp = dp.tolist()
        dp = [math.nan if x < -20000 else x for x in dp]
        # dp = dp[dp > -10000].astype(int)

        vpg_fp = get_vpg_fiducials(signal, onsets=onsets)
        u = vpg_fp["u"].values.astype(int)
        v = vpg_fp["v"].values.astype(int)

        # Plot to check that the fiducials are correct
        # plt.figure(figsize=(15, 5))
        # plt.plot(signal, label="Signal")
        # plt.plot(peaks, signal[peaks], "bo", label="Peaks")
        # plt.plot(onsets, signal[onsets], "ro", label="Onsets")
        # plt.plot(dn, signal[dn], "go", label="Dicrotic Notch")
        # plt.plot(dp, signal[dp], "mo", label="Diastolic Peak")

        fiducials_diag = pd.DataFrame(
            {
                "onset": np.array(onsets)[:-1],
                "offset": np.array(onsets)[1:],
                "sp": peaks,
                "dn": dn,
                "u": u,
                "v": v,
                "a": a,
                "b": b,
                "dp": dp,
            }
        )
        fiducials_diag.dropna(inplace=True)
        fiducials_diag = fiducials_diag.astype(int)

        Tpi = fiducials_diag["offset"] / fs - fiducials_diag["onset"] / fs
        Tsys = fiducials_diag["dn"] / fs - fiducials_diag["onset"] / fs
        Tdia = fiducials_diag["offset"] / fs - fiducials_diag["dn"] / fs
        Tsp = fiducials_diag["sp"] / fs - fiducials_diag["onset"] / fs
        Tdp = fiducials_diag["dp"] / fs - fiducials_diag["onset"] / fs
        dT = fiducials_diag["dp"] / fs - fiducials_diag["sp"] / fs

        # ---- Amplitudes -----
        # Asp: difference in amplitude between onset and sp
        # Adn: difference in amplitude between onset and dn
        # Adp: difference in amplitude between onset and dp
        Asp = signal[fiducials_diag["sp"]] - signal[fiducials_diag["onset"]]
        Adn = signal[fiducials_diag["dn"]] - signal[fiducials_diag["onset"]]
        Adp = signal[fiducials_diag["dp"]] - signal[fiducials_diag["onset"]]

        # ---- Areas -----
        # AUCpi: area under the whole pulse (between onset and offset)
        # AUCsys: area under pulse wave between onset and dn
        # AUCdia: area under pulse wave between dn and offset
        AUCpi = np.zeros(len(fiducials_diag))
        AUCsys = np.zeros(len(fiducials_diag))
        AUCdia = np.zeros(len(fiducials_diag))
        for i in range(len(fiducials_diag)):
            AUCpi[i] = np.trapz(
                signal[
                    fiducials_diag["onset"].iloc[i] : fiducials_diag["offset"].iloc[i]
                ]
            )
            AUCsys[i] = np.trapz(
                signal[fiducials_diag["onset"].iloc[i] : fiducials_diag["dn"].iloc[i]]
            )
            AUCdia[i] = np.trapz(
                signal[fiducials_diag["dn"].iloc[i] : fiducials_diag["offset"].iloc[i]]
            )

        # ---- Ratios -----
        # TsysTdia: ratio between Tsys and Tdia
        # TspTpi: ratio between Tsp and Tpi
        # AdpAsp: ratio between Adp and Asp --> reflection index!
        # IPA: ratio of the area under diastolic curve and the area under systolic curve
        # TspAsp: ratio between Tsp and Asp
        # AspdT: ratio between Asp and dT --> Stifness index
        # AspTpiTsp: ratio between Asp and (Tpi - Tsp)
        TsysTdia = Tsys / Tdia
        TspTpi = Tsp / Tpi
        AdpAsp = Adp / Asp
        IPA = AUCdia / AUCsys
        TspAsp = Tsp / Asp
        AspdT = Asp / dT
        AspTpiTsp = Asp / (Tpi - Tsp)

        # ---- Derivatives -----
        # Tu: time between the pulse onset and u
        # Tv: time between the pulse onset and v
        # Ta: time between the pulse onset and a
        # Tb: time between the pulse onset and b
        Tu = fiducials_diag["u"] / fs - fiducials_diag["onset"] / fs
        Tv = fiducials_diag["v"] / fs - fiducials_diag["onset"] / fs
        Ta = fiducials_diag["a"] / fs - fiducials_diag["onset"] / fs
        Tb = fiducials_diag["b"] / fs - fiducials_diag["onset"] / fs

        features_morph = pd.DataFrame(
            {
                "mean_rr": mean_rr,
                "max_rr": max_rr,
                "min_rr": min_rr,
                "pwa": pwa,
                "Tpi": np.mean(Tpi),
                "Tsys": np.mean(Tsys),
                "Tdia": np.mean(Tdia),  # "Tsp": np.mean(Tsp), #"Tdp": np.mean(Tdp),
                # "dT": np.mean(dT),
                "Asp": np.mean(Asp),
                "Adn": np.mean(Adn),
                # "Adp": np.mean(Adp),
                "AUCpi": np.mean(AUCpi),
                "AUCsys": np.mean(AUCsys),
                "AUCdia": np.mean(AUCdia),
                "Tsys/Tdia": np.mean(TsysTdia),
                "Tsp/Tpi": np.mean(TspTpi),  # "Adp/Asp": np.mean(AdpAsp),
                "IPA": np.mean(IPA),
                "Tsp/Asp": np.mean(TspAsp),
                # "Asp/dT": np.mean(AspdT),
                "Asp/(Tpi-Tsp)": np.mean(AspTpiTsp),
                "Tu": np.mean(Tu),
                "Tv": np.mean(Tv),
                "Ta": np.mean(Ta),
                "Tb": np.mean(Tb),
            },
            index=[0],
        )


        # HRV features with neurokit
        hrv_time = nk.hrv_time(trial_peaks[col], sampling_rate=100, show=False)

        # hrv_nonlinear = nk.hrv_nonlinear(trial_peaks[col], sampling_rate=100, show=False)

        features_dict[col] = pd.concat([features_morph, hrv_time], axis = 1)

        # features_dict[col] = features_morph

    # Final summary DataFrame
    features_df_morph_sub = pd.concat(features_dict).droplevel(1)
    features_df_morph_sub["label"] = [
        condition.split("_")[1] for condition in features_df_morph_sub.index
    ]
    features_df_morph_sub["subject_id"] = sub.split(".")[0]

    col_upper = col.upper()
    if "BASELINE" in col_upper:
        label = 0
    elif "LOW" in col_upper:
        label = 1
    elif "HIGH" in col_upper:
        label = 2
    elif "REST" in col_upper:
        label = 3
    else:
        label = -1  # unknown or unexpected pattern

    # Append the features DataFrame for this subject to the main features DataFrame
    features_df_morph = pd.concat(
        [features_df_morph, features_df_morph_sub], ignore_index=True
    )

features_df_morph = features_df_morph[features_df_morph["label"] != "Baseline"]

features_df_morph.drop(columns = ["HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2", "HRV_SDANN5", "HRV_SDNNI5"], inplace = True)

# EDA features
# MS's path
# eda_path = "/Users/augenpro/Documents/GitHub/AI4PAIN_challenge/AI4Pain 2025 Dataset-20250414T153814Z-001/AI4Pain 2025 Dataset/train_validation/eda/"

# SM's path
eda_path = "C:/Users/serena.moscato4/OneDrive - Alma Mater Studiorum Università di Bologna/Personal_Health_Systems_Lab/AI4PAIN_challenge/AI4Pain 2025 Dataset-20250414T153814Z-001/AI4Pain 2025 Dataset/train_validation/Eda"

subjects = sorted(os.listdir(eda_path))
all_features_eda = []
all_dfs_eda = []

for i, sub in enumerate(subjects):
    subject_csv_eda = pd.read_csv(os.path.join(eda_path, sub), header=0)
    print(f"Processing subject {i + 1}/{len(subjects)}: {sub}")

    for col in subject_csv_eda.columns:
        values_eda = subject_csv_eda[col].dropna().values
        filtered_values_eda = nk.eda_clean(values_eda, sampling_rate=fs)

        col_upper = col.upper()
        if "BASELINE" in col_upper or "REST" in col_upper:
            # From the BASELINE/REST period, take the last 10 seconds
            filtered_values_eda = filtered_values_eda[-10 * fs :]

        signals_eda, info = nk.eda_process(filtered_values_eda, sampling_rate=fs)
        features_eda = nk.eda_analyze(
            signals_eda, sampling_rate=fs, method="interval-related"
        )

        features_craft_eda = EDA_features(signals_eda["EDA_Raw"], fs)
        features_statistics_eda = statistic_feature(
            np.array(signals_eda["EDA_Clean"]), fs
        )

        all_features_eda = pd.concat(
            [features_eda, features_craft_eda, features_statistics_eda], axis=1
        )

        # Determine label from column name
        # col_upper = col.upper()
        # if "BASELINE" in col_upper:
        #     label = 0
        # elif "LOW" in col_upper:
        #     label = 1
        # elif "HIGH" in col_upper:
        #     label = 2
        # elif "REST" in col_upper:
        #     label = 3
        # else:
        #     label = -1  # unknown or unexpected pattern

        all_features_eda["subject_id"] = sub.split(".")[0]
        all_features_eda["label"] = col.split("_")[1] if "_" in col else "unknown"

        all_dfs_eda.append(all_features_eda)

final_eda_df = pd.concat(all_dfs_eda, ignore_index=True)
final_eda_df = final_eda_df[final_eda_df["label"] != "Baseline"]

# final_eda_df.drop(columns = ["EDA_Sympathetic", "EDA_SympatheticN", "EDA_Autocorrelation"], inplace = True)

# Resp featuers
# MS's path
# resp_path = "/Users/augenpro/Documents/GitHub/AI4PAIN_challenge/AI4Pain 2025 Dataset-20250414T153814Z-001/AI4Pain 2025 Dataset/train_validation/resp/"

# SM's path
resp_path = "C:/Users/serena.moscato4/OneDrive - Alma Mater Studiorum Università di Bologna/Personal_Health_Systems_Lab/AI4PAIN_challenge/AI4Pain 2025 Dataset-20250414T153814Z-001/AI4Pain 2025 Dataset/train_validation/Resp"

subjects = sorted(os.listdir(resp_path))

fs = 100

features_df_resp = pd.DataFrame()

for i, sub in enumerate(subjects):
    print(f"Processing subject {i+1}/{len(subjects)}: {sub}")
    resp = pd.read_csv(os.path.join(resp_path, sub), header=0)
    original_lengths = resp.notna().sum().to_dict()

    # Concatenate all columns and drop NaN values
    resp_full = np.concatenate([resp[col].dropna().values for col in resp.columns])
    # plt.figure(figsize= (12,8))
    # plt.plot(resp_full)
    # resp_full_filtered = nk.rsp_clean(resp_full, sampling_rate = 100, method = "BioSPPy")
    # plt.plot(resp_full_filtered)
    signals, info = nk.rsp_process(resp_full, sampling_rate=100)

    # Reshape the filtered data back to the original columns
    start = 0
    for trial_name, length in original_lengths.items():
        end = start + length
        trial_signals = signals.iloc[start:end]

        col_upper = trial_name.upper()
        if "BASELINE" in col_upper or "REST" in col_upper:
            # From the BASELINE/REST period, take the last 10 seconds
            trial_signals = trial_signals.iloc[-10 * fs :]

        # Compute mean for each feature in this trial
        feature_trial = pd.DataFrame({
            "RSP_n_peaks": trial_signals["RSP_Peaks"].sum(),
            "RSP_Rate": trial_signals["RSP_Rate"].mean(),
            "RSP_Amplitude": trial_signals["RSP_Amplitude"].mean(),
            "RSP_RVT": trial_signals["RSP_RVT"].mean(),
            "RSP_Symmetry_RiseDecay": trial_signals["RSP_Symmetry_RiseDecay"].mean(),
            "RSP_Symmetry_PeakTrough": trial_signals["RSP_Symmetry_PeakTrough"].mean()
        }, index = [0])

        feature_trial["label"] = trial_name.split("_")[1] if "_" in trial_name else "unknown"
        feature_trial["subject_id"] = sub.replace(".csv", "")

        features_df_resp = pd.concat([features_df_resp, feature_trial], ignore_index=True)

        start = end  # Move to next trial

features_df_resp = features_df_resp[features_df_resp["label"] != "Baseline"]

features_df_resp.shape, final_eda_df.shape, features_df_morph.shape

final_eda_df.drop(columns = ["label","subject_id"], inplace=True)
features_df_resp.drop(columns = ["label", "subject_id"], inplace = True)

final_df = pd.concat([features_df_morph, final_eda_df, features_df_resp], axis = 1)
final_df.to_csv("final_df.csv", index=False)

# Visualization
sns.set_context("talk")
sns.set_palette("Set1")
feature = "mean_rr"
# Boxplot
plt.figure(figsize=(10,8))
sns.boxplot(data=final_df, x="label", y=feature)
plt.title(feature)
plt.xlabel("label")
# plt.ylabel("Tsp/Asp")
plt.xticks(rotation=45)
plt.tight_layout()

# Correlation matrix between all features
plt.figure(figsize=(20, 20))
corr = final_df.drop(columns=["label", "subject_id"]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False, fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation matrix")
plt.tight_layout()
