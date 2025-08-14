#!/usr/bin/env python
# coding: utf-8

# In[41]:


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

bvp_folder_val = 'AI4Pain_2025_Dataset/validation/Bvp'
eda_folder_val = 'AI4Pain_2025_Dataset/validation/Eda'
resp_folder_val = 'AI4Pain_2025_Dataset/validation/Resp'
spo_folder_val = 'AI4Pain_2025_Dataset/validation/SpO2'


# In[47]:


#Data normalization - this is a global normalization (i.e., not normalized using the baseline mean and sd)
for i in range(list_train.shape[0]):
    subject_id = str(list_train.loc[i, 'Id'])  #change here to normalize validation data - if using validation data make sure to add '_val' to the signal folder
    input_csv_path = os.path.join(bvp_folder, subject_id + '.csv') #change bvp_folder to  normalize other signals
    
    
    subject_csv = pd.read_csv(input_csv_path, header=0)

    # Flatten all values to 1D
    flat_values = subject_csv.values.flatten()
    
    # Global z-score normalization 
    mean = np.nanmean(flat_values)
    std = np.nanstd(flat_values)
    normalized_flat = (flat_values - mean) / std

    #bring them back to original columns
    normalized_values = normalized_flat.reshape(subject_csv.shape)
    normalized_df = pd.DataFrame(normalized_values, columns=subject_csv.columns)

    
    normalized_csv_path = os.path.join(bvp_folder, subject_id + '_normalized.csv')
    normalized_df.to_csv(normalized_csv_path, index=False)
