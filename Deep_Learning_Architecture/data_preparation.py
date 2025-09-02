#!/usr/bin/env python
# coding: utf-8

# In[41]:
#this code is rearranging the data by converting the columns in row, in order to create the dataset for the DL network.
#normalization is done before this
#then, the code split the baseline into 6 chunks (we're using baseline as no pain for now), rest sequence is removed and 
#only the first 9 seconds (first 900 values) are retained for all the conditions
#the final shape in this case will be:
#training: number of rows: 12 HIGH * 41 subjects + 12 LOW * 41 subjects + 6 Nopain (Baseline) for 41 subjects, n of columns: 900
#same applies to the validation data

import pandas as pd
import os
import numpy as np


# In[43]:


list_train=pd.read_csv('list_sub_train.csv', header=0)
list_val=pd.read_csv('list_sub_val.csv', header=0)
list_train.head()


# In[45]:


bvp_folder = 'AI4Pain_2025_Dataset/train/Bvp'
eda_folder = 'AI4Pain_2025_Dataset/train/Eda'
resp_folder = 'AI4Pain_2025_Dataset/train/Resp'
spo_folder = 'AI4Pain_2025_Dataset/train/SpO2'


# In[47]:


#preparing bvp data for training
all_data = []
all_labels = []
all_column_names = []
all_subject_ids = []

for i in range(list_train.shape[0]):
    subject_id = str(list_train.loc[i, 'Id'])
    subject_csv = pd.read_csv(os.path.join(bvp_folder, subject_id + '.csv'), header=0) #add _normalized before '.csv' to use normalized data

    for col in subject_csv.columns:
        values = subject_csv[col].values
        all_data.append(values)

        # Determine label from column name
        col_upper = col.upper()
        if 'BASELINE' in col_upper:
            label = 0
        elif 'LOW' in col_upper:
            label = 1
        elif 'HIGH' in col_upper:
            label = 2
        elif 'REST' in col_upper:
            label = 3
        else:
            label = -1  # unknown or unexpected pattern

        all_labels.append(label)
        all_column_names.append(col)  # Use the actual column name
        all_subject_ids.append(subject_id)

# Find max length
max_len = max(len(row) for row in all_data)

# Pad each row to max length with NaNs
padded_data = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in all_data]
column_names = [f"{i+1:03d}" for i in range(max_len)]

# Build final DataFrame
df = pd.DataFrame(padded_data, columns=column_names)
df['label'] = all_labels
df['column_name'] = all_column_names
df['subject_id'] = all_subject_ids

df.to_csv('training_dataset_bvp.csv', index=False)


# In[48]:


#preparing eda data for training
all_data = []
all_labels = []
all_column_names = []
all_subject_ids = []

for i in range(list_train.shape[0]):
    subject_id = str(list_train.loc[i, 'Id'])
    subject_csv = pd.read_csv(os.path.join(eda_folder, subject_id + '.csv'), header=0)

    for col in subject_csv.columns:
        values = subject_csv[col].values
        all_data.append(values)

        # Determine label from column name
        col_upper = col.upper()
        if 'BASELINE' in col_upper:
            label = 0
        elif 'LOW' in col_upper:
            label = 1
        elif 'HIGH' in col_upper:
            label = 2
        elif 'REST' in col_upper:
            label = 3
        else:
            label = -1  # unknown or unexpected pattern

        all_labels.append(label)
        all_column_names.append(col)  # Use the actual column name
        all_subject_ids.append(subject_id)

# Find max length
max_len = max(len(row) for row in all_data)

# Pad each row to max length with NaNs
padded_data = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in all_data]
column_names = [f"{i+1:03d}" for i in range(max_len)]

# Build final DataFrame
df = pd.DataFrame(padded_data, columns=column_names)
df['label'] = all_labels
df['column_name'] = all_column_names
df['subject_id'] = all_subject_ids

df.to_csv('training_dataset_eda.csv', index=False)


# In[49]:


#preparing resp data for training
all_data = []
all_labels = []
all_column_names = []
all_subject_ids = []

for i in range(list_train.shape[0]):
    subject_id = str(list_train.loc[i, 'Id'])
    subject_csv = pd.read_csv(os.path.join(resp_folder, subject_id + '.csv'), header=0)

    for col in subject_csv.columns:
        values = subject_csv[col].values
        all_data.append(values)

        # Determine label from column name
        col_upper = col.upper()
        if 'BASELINE' in col_upper:
            label = 0
        elif 'LOW' in col_upper:
            label = 1
        elif 'HIGH' in col_upper:
            label = 2
        elif 'REST' in col_upper:
            label = 3
        else:
            label = -1  # unknown or unexpected pattern

        all_labels.append(label)
        all_column_names.append(col)  # Use the actual column name
        all_subject_ids.append(subject_id)

# Find max length
max_len = max(len(row) for row in all_data)

# Pad each row to max length with NaNs
padded_data = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in all_data]
column_names = [f"{i+1:03d}" for i in range(max_len)]

# Build final DataFrame
df = pd.DataFrame(padded_data, columns=column_names)
df['label'] = all_labels
df['column_name'] = all_column_names
df['subject_id'] = all_subject_ids

df.to_csv('training_dataset_resp.csv', index=False)


# In[50]:


#preparing SpO2 data for training
all_data = []
all_labels = []
all_column_names = []
all_subject_ids = []

for i in range(list_train.shape[0]):
    subject_id = str(list_train.loc[i, 'Id'])
    subject_csv = pd.read_csv(os.path.join(spo_folder, subject_id + '.csv'), header=0)

    for col in subject_csv.columns:
        values = subject_csv[col].values
        all_data.append(values)

        # Determine label from column name
        col_upper = col.upper()
        if 'BASELINE' in col_upper:
            label = 0
        elif 'LOW' in col_upper:
            label = 1
        elif 'HIGH' in col_upper:
            label = 2
        elif 'REST' in col_upper:
            label = 3
        else:
            label = -1  # unknown or unexpected pattern

        all_labels.append(label)
        all_column_names.append(col)  # Use the actual column name
        all_subject_ids.append(subject_id)

# Find max length
max_len = max(len(row) for row in all_data)

# Pad each row to max length with NaNs
padded_data = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in all_data]
column_names = [f"{i+1:03d}" for i in range(max_len)]

# Build final DataFrame
df = pd.DataFrame(padded_data, columns=column_names)
df['label'] = all_labels
df['column_name'] = all_column_names
df['subject_id'] = all_subject_ids

df.to_csv('training_dataset_spo.csv', index=False)


# In[51]:


bvp_folder_val = 'AI4Pain_2025_Dataset/validation/Bvp'
eda_folder_val = 'AI4Pain_2025_Dataset/validation/Eda'
resp_folder_val = 'AI4Pain_2025_Dataset/validation/Resp'
spo_folder_val = 'AI4Pain_2025_Dataset/validation/SpO2'


# In[52]:


#preparing bvp data for validation
all_data = []
all_labels = []
all_column_names = []
all_subject_ids = []

for i in range(list_val.shape[0]):
    subject_id = str(list_val.loc[i, 'Id'])
    subject_csv = pd.read_csv(os.path.join(bvp_folder_val, subject_id + '.csv'), header=0)

    for col in subject_csv.columns:
        values = subject_csv[col].values
        all_data.append(values)

        # Determine label from column name
        col_upper = col.upper()
        if 'BASELINE' in col_upper:
            label = 0
        elif 'LOW' in col_upper:
            label = 1
        elif 'HIGH' in col_upper:
            label = 2
        elif 'REST' in col_upper:
            label = 3
        else:
            label = -1  # unknown or unexpected pattern

        all_labels.append(label)
        all_column_names.append(col)  # Use the actual column name
        all_subject_ids.append(subject_id)

# Find max length
max_len = max(len(row) for row in all_data)

# Pad each row to max length with NaNs
padded_data = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in all_data]
column_names = [f"{i+1:03d}" for i in range(max_len)]

# Build final DataFrame
df = pd.DataFrame(padded_data, columns=column_names)
df['label'] = all_labels
df['column_name'] = all_column_names
df['subject_id'] = all_subject_ids

df.to_csv('validation_dataset_bvp.csv', index=False)


# In[54]:


#preparing eda data for validation
all_data = []
all_labels = []
all_column_names = []
all_subject_ids = []

for i in range(list_val.shape[0]):
    subject_id = str(list_val.loc[i, 'Id'])
    subject_csv = pd.read_csv(os.path.join(eda_folder_val, subject_id + '.csv'), header=0)

    for col in subject_csv.columns:
        values = subject_csv[col].values
        all_data.append(values)

        # Determine label from column name
        col_upper = col.upper()
        if 'BASELINE' in col_upper:
            label = 0
        elif 'LOW' in col_upper:
            label = 1
        elif 'HIGH' in col_upper:
            label = 2
        elif 'REST' in col_upper:
            label = 3
        else:
            label = -1  # unknown or unexpected pattern

        all_labels.append(label)
        all_column_names.append(col)  # Use the actual column name
        all_subject_ids.append(subject_id)

# Find max length
max_len = max(len(row) for row in all_data)

# Pad each row to max length with NaNs
padded_data = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in all_data]
column_names = [f"{i+1:03d}" for i in range(max_len)]

# Build final DataFrame
df = pd.DataFrame(padded_data, columns=column_names)
df['label'] = all_labels
df['column_name'] = all_column_names
df['subject_id'] = all_subject_ids

df.to_csv('validation_dataset_eda.csv', index=False)


# In[55]:


#preparing resp data for validation
all_data = []
all_labels = []
all_column_names = []
all_subject_ids = []

for i in range(list_val.shape[0]):
    subject_id = str(list_val.loc[i, 'Id'])
    subject_csv = pd.read_csv(os.path.join(resp_folder_val, subject_id + '.csv'), header=0)

    for col in subject_csv.columns:
        values = subject_csv[col].values
        all_data.append(values)

        # Determine label from column name
        col_upper = col.upper()
        if 'BASELINE' in col_upper:
            label = 0
        elif 'LOW' in col_upper:
            label = 1
        elif 'HIGH' in col_upper:
            label = 2
        elif 'REST' in col_upper:
            label = 3
        else:
            label = -1  # unknown or unexpected pattern

        all_labels.append(label)
        all_column_names.append(col)  # Use the actual column name
        all_subject_ids.append(subject_id)

# Find max length
max_len = max(len(row) for row in all_data)

# Pad each row to max length with NaNs
padded_data = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in all_data]
column_names = [f"{i+1:03d}" for i in range(max_len)]

# Build final DataFrame
df = pd.DataFrame(padded_data, columns=column_names)
df['label'] = all_labels
df['column_name'] = all_column_names
df['subject_id'] = all_subject_ids

df.to_csv('validation_dataset_resp.csv', index=False)


# In[57]:


#preparing SpO2 data for validation
all_data = []
all_labels = []
all_column_names = []
all_subject_ids = []

for i in range(list_val.shape[0]):
    subject_id = str(list_val.loc[i, 'Id'])
    subject_csv = pd.read_csv(os.path.join(spo_folder_val, subject_id + '.csv'), header=0)

    for col in subject_csv.columns:
        values = subject_csv[col].values
        all_data.append(values)

        # Determine label from column name
        col_upper = col.upper()
        if 'BASELINE' in col_upper:
            label = 0
        elif 'LOW' in col_upper:
            label = 1
        elif 'HIGH' in col_upper:
            label = 2
        elif 'REST' in col_upper:
            label = 3
        else:
            label = -1  # unknown or unexpected pattern

        all_labels.append(label)
        all_column_names.append(col)  # Use the actual column name
        all_subject_ids.append(subject_id)

# Find max length
max_len = max(len(row) for row in all_data)

# Pad each row to max length with NaNs
padded_data = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in all_data]
column_names = [f"{i+1:03d}" for i in range(max_len)]

# Build final DataFrame
df = pd.DataFrame(padded_data, columns=column_names)
df['label'] = all_labels
df['column_name'] = all_column_names
df['subject_id'] = all_subject_ids

df.to_csv('validation_dataset_spo.csv', index=False)

#-------------------------------------------------------------#

#splitting baseline data into 6 pieces (since 9 seconds will be used for low-high pain conditions due to 
#length of the other conditions, only the first 9*6=54 seconds will be retained


# Load the full DataFrame
df = pd.read_csv('training_dataset_bvp_normalized.csv') #to change for validation

# Extract just the signal columns (excluding label, column_name, subject_id)
signal_columns = [col for col in df.columns if col.isdigit()]

# New rows will be collected here
new_rows = []

for _, row in df.iterrows():
    if 'Baseline' in row['column_name']:
        
        values = row[signal_columns].dropna().values[:5400]  # First 5400 values only

        # Check we actually have 5400 values to split
        if len(values) < 5400:
            continue  # Skip if not enough data

        # Split into 6 chunks of 900
        chunks = [values[i*900:(i+1)*900] for i in range(6)]

        for idx, chunk in enumerate(chunks):
            padded_chunk = list(chunk) + [float('nan')] * (len(signal_columns) - 900)
            new_row = dict(zip(signal_columns, padded_chunk))
            new_row['label'] = row['label']
            new_row['column_name'] = f"{row['column_name']}_part{idx+1}"
            new_row['subject_id'] = row['subject_id']
            new_rows.append(new_row)
    else:
        # Keep other rows unchanged
        new_rows.append(row)


df_updated = pd.DataFrame(new_rows)

# Save it
df_updated.to_csv('training_dataset_bvp_normalized_baseline_splitted.csv', index=False) #change for validation

#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#
#removing REST condition and cut all the conditions to 900 timepoints
import pandas as pd

# Load the dataset
df = pd.read_csv('training_dataset_bvp_normalized_baseline_splitted.csv')
print(f"Initial shape: {df.shape}")

# Step 1: Remove rows with 'REST' in the 'column_name'
df_no_rest = df[~df['column_name'].str.upper().str.contains('REST')].reset_index(drop=True)
print(f"After removing 'REST' rows: {df_no_rest.shape}")

# Step 2: Keep only the first 900 signal columns
# Identify signal columns (numerical column names)
signal_columns = [col for col in df_no_rest.columns if col.isdigit()]
signal_columns_900 = signal_columns[:900]

# Final columns to keep
final_columns = signal_columns_900 + ['label', 'column_name', 'subject_id']
df_trimmed = df_no_rest[final_columns]
print(f"After trimming to first 900 values per row: {df_trimmed.shape}")

# Save the result
df_trimmed.to_csv('training_dataset_bvp_ready_for_DL.csv', index=False)
