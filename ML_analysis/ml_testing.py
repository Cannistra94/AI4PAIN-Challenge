import pandas as pd
import joblib
import os

# Load test data
test_df = pd.read_csv("final_df_test.csv")
test_df.drop(columns = ["Tsys/Tdia", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD", "EDA_SIE"], inplace=True)

# Extract metadata
subject_ids = test_df["subject_id"].values
sample_ids = test_df["label"].values
X_test = test_df.drop(columns=["subject_id", "label"])

# Label mapping back to original strings
label_map = {0: "No_Pain", 1: "Low_Pain", 2: "High_Pain"}

# Load top 5 configs
top_5 = pd.read_csv("top_5_model_configs.csv").head(5)

# Predict and save for each model
for _, row in top_5.iterrows():
    model_name = row["Model"]
    k = int(row["Top_K_Features"])
    model_id = f"{model_name.replace(' ', '_')}_top{k}_FULL"
    model_path = f"retrained_models/{model_id}.pkl"

    print(f"\nLoading model: {model_path}")
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)
    y_pred_label = [label_map[i] for i in y_pred]

    submission_df = pd.DataFrame({
        "Subject_ID": subject_ids,
        "Sample": sample_ids,
        "Predicted_label": y_pred_label
    })

    submission_file = f"submission_{model_id}.csv"
    submission_df.to_csv(submission_file, index=False)
    print(f"Saved: {submission_file}")
