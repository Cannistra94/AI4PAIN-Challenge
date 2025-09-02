import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif  # or mutual_info_classif
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import json
import os
from sklearn.pipeline import Pipeline

final_df = pd.read_csv("final_df.csv")

final_df.drop(columns = ["Tsys/Tdia", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD", "EDA_SIE"], inplace=True)

final_df['label'] = final_df['label'].replace({
    'REST': 0,
    'LOW': 1,
    'HIGH': 2
})

# Split features and target
X = final_df.drop(columns=["label", "subject_id"])
y = final_df["label"]
groups = final_df["subject_id"]

# Initialize scaler and models
scaler = StandardScaler()

# Setup models

param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"]  # you can also try "liblinear" for smaller datasets
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "gamma": ['scale', 'auto'],
        "kernel": ['rbf']  # you can add 'poly' or 'sigmoid' if desired
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1]
    }
}

# Models
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Stratified Group CV
cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

# Storage
all_results = []
summary_class_metrics = []
selected_features_per_fold = []
top_k_features_list = [10, 20, 30, 40, 54] # Insert the real number of features after insertion of morphological and RESP features

os.makedirs("saved_models", exist_ok=True)
top_models = []

for k in top_k_features_list:
    print(f"\n=== Starting feature selection with Top {k} features ===")
    for name, base_model in base_models.items():
        print(f"\n--- Model: {name} ---")
        fold_accuracies, fold_sensitivities, fold_specificities = [], [], []
        fold_precisions, fold_f1s = [], []
        per_class_scores = {label: {"precision": [], "recall": [], "f1": [], "support": []} for label in [0, 1, 2]}

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), 1):
            print(f"Processing Fold {fold_idx}...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)

            selected_feature_names = X.columns[selector.get_support()].tolist()
            selected_features_per_fold.append({
                "Model": name, "Top_K_Features": k, "Fold": fold_idx,
                "Selected_Features": selected_feature_names
            })

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            print("Running GridSearchCV...")
            grid = GridSearchCV(base_model, param_grids[name], cv=10, scoring='f1_macro')
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test_scaled)

            print("Fold completed. Evaluating metrics...")
            acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(acc)

            recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
            precisions = precision_score(y_test, y_pred, average=None, zero_division=0)
            f1s = f1_score(y_test, y_pred, average=None, zero_division=0)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            specificities = []
            for i in range(3):
                TP = cm[i, i]
                FN = cm[i, :].sum() - TP
                FP = cm[:, i].sum() - TP
                TN = cm.sum() - (TP + FP + FN)
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                specificities.append(specificity)

            fold_sensitivities.append(np.mean(recalls))
            fold_specificities.append(np.mean(specificities))
            fold_precisions.append(np.mean(precisions))
            fold_f1s.append(np.mean(f1s))

            for i in range(3):
                per_class_scores[i]["precision"].append(precisions[i])
                per_class_scores[i]["recall"].append(recalls[i])
                per_class_scores[i]["f1"].append(f1s[i])
                per_class_scores[i]["support"].append((y_test == i).sum())

            pipeline = Pipeline([
                ("selector", selector),
                ("scaler", scaler),
                ("model", best_model)
            ])

        f1_macro = np.mean(f1s)
        model_id = f"{name.replace(' ', '_')}_top{k}_fold{fold_idx}"
        model_path = f"saved_models/{model_id}.pkl"
        meta_path = f"saved_models/{model_id}_meta.json"

        joblib.dump(pipeline, model_path)

        metadata = {
            "model_name": name,
            "top_k_features": k,
            "fold": fold_idx,
            "selected_features": selected_feature_names,
            "best_params": grid.best_params_,
            "metrics":{
                "accuracy": acc,
                "mean_f1": f1_macro,
                "precision": np.mean(precisions),
                "recall": np.mean(recalls),
                "specificity": np.mean(specificities)
            }
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Collect for top 5
        top_models.append({
            "model_id": model_id,
            "mean_f1": f1_macro,
            "model_path": model_path,
            "meta_path": meta_path
        })

        all_results.append({
            "Model": name,
            "Top_K_Features": k,
            "Accuracy": np.mean(fold_accuracies),
            "Accuracy_Std": np.std(fold_accuracies),
            "Sensitivity": np.mean(fold_sensitivities),
            "Sensitivity_Std": np.std(fold_sensitivities),
            "Specificity": np.mean(fold_specificities),
            "Specificity_Std": np.std(fold_specificities),
            "Precision": np.mean(fold_precisions),
            "Precision_Std": np.std(fold_precisions),
            "F1_Score": np.mean(fold_f1s),
            "F1_Score_Std": np.std(fold_f1s)
        })

        for label in [0, 1, 2]:
            summary_class_metrics.append({
                "Model": name,
                "Top_K_Features": k,
                "Class": f"Class {label}",
                "Precision_Mean": np.mean(per_class_scores[label]["precision"]),
                "Precision_Std": np.std(per_class_scores[label]["precision"]),
                "Recall_Mean": np.mean(per_class_scores[label]["recall"]),
                "Recall_Std": np.std(per_class_scores[label]["recall"]),
                "F1_Mean": np.mean(per_class_scores[label]["f1"]),
                "F1_Std": np.std(per_class_scores[label]["f1"]),
                "Support_Mean": np.mean(per_class_scores[label]["support"])
            })
        print(f"Finished all folds for Model: {name} with Top {k} features.")

# Convert to DataFrames
results_df = pd.DataFrame(all_results)
class_report_df = pd.DataFrame(summary_class_metrics)
selected_features_df = pd.DataFrame(selected_features_per_fold)

# Save (optional)
results_df.to_csv("overall_model_metrics_opt_10CV_inner10CV_savedmodels.csv", index=False)
class_report_df.to_csv("class_wise_metrics_opt_10CV_inner10CV_savedmodels.csv", index=False)
selected_features_df.to_csv("selected_features_per_fold_opt_10CV_inner10CV_savedmodels.csv", index=False)

top_models_sorted = sorted(top_models, key = lambda x: x["mean_f1"], reverse=True)[:5]

print("\nTop 5 Models by Mean F1 Score:")
for model in top_models_sorted:
    print(f"{model['model_id']}: {model['mean_f1']:.4f}")
