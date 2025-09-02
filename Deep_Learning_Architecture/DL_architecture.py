#the script builds, trains, and evaluates a deep learning model that combines residual CNN blocks, 
#attention, and LSTMs to classify multimodal physiological time-series data (BVP, EDA, RESP) into three classes,
#reporting performance with accuracy, precision, recall, F1-scores, and specificity.

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Load CSVs
#--------------------------for single modality data use the following---------------------------------#
train_df = pd.read_csv('training_dataset_bvp_ready_for_DL.csv')
val_df = pd.read_csv('validation_dataset_bvp_ready_for_DL.csv')

print(f"Training shape before: {train_df.shape}, Validation shape before: {val_df.shape}")

# drop columns
X_train = train_df.drop(columns=['label', 'column_name', 'subject_id'], errors='ignore').values
X_val = val_df.drop(columns=['label', 'column_name', 'subject_id'], errors='ignore').values
y_train = train_df['label'].values
y_val = val_df['label'].values

print(f"Training shape after: {X_train.shape}, Validation shape after: {X_val.shape}")

#--------------------------for multimodal data use the following---------------------------------#
# training
train_df1 = pd.read_csv('training_dataset_bvp_ready_for_DL.csv')
train_df2 = pd.read_csv('training_dataset_eda_ready_for_DL.csv')
train_df3 = pd.read_csv('training_dataset_resp_ready_for_DL.csv')

# validation
val_df1 = pd.read_csv('validation_dataset_bvp_ready_for_DL.csv')
val_df2 = pd.read_csv('validation_dataset_eda_ready_for_DL.csv')
val_df3 = pd.read_csv('validation_dataset_resp_ready_for_DL.csv')

# Columns to drop
drop_cols = ['label', 'column_name', 'subject_id']

# Extract features and drop columns
X_train1 = train_df1.drop(columns=drop_cols, errors='ignore').values
X_train2 = train_df2.drop(columns=drop_cols, errors='ignore').values
X_train3 = train_df3.drop(columns=drop_cols, errors='ignore').values

X_val1 = val_df1.drop(columns=drop_cols, errors='ignore').values
X_val2 = val_df2.drop(columns=drop_cols, errors='ignore').values
X_val3 = val_df3.drop(columns=drop_cols, errors='ignore').values

# ensure consistency in shapes
assert X_train1.shape == X_train2.shape == X_train3.shape, "Train sets must match in shape"
assert X_val1.shape == X_val2.shape == X_val3.shape, "Validation sets must match in shape"

# stack the 3 feature arrays -> final shape: (samples, features, 3) 
X_train = np.stack([X_train1, X_train2, X_train3], axis=-1)
X_val = np.stack([X_val1, X_val2, X_val3], axis=-1)

# labels (they're consistent across csvs so we can extract labels from the first csv
y_train = train_df1['label'].values
y_val = val_df1['label'].values

# shapes
print("Training shape:", X_train.shape)   
print("Validation shape:", X_val.shape) 

#-------------------------------------------------------------------#

# One-hot encode labels - needed for multi-class classification tasks as the model will expect labels in this format, 
#since it'll output probability distribution:

#[1, 0, 0] → class 0  
#[0, 1, 0] → class 1
#[0, 0, 1] → class 2 

y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)


#defining model--this is the best performing model for the multimodal dataset so far (accuracy around 65%)
def residual_block(x, filters, kernel_size, pool_size=2):
    shortcut = x
    # Project shortcut to match number of filters
    if x.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding='same')(shortcut)

    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(0.3)(x)
    return x


def create_model():
    inputs = Input(shape=(900, 3))

    #  Conv layer - pooling - dropout
    x = Conv1D(32, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # residual blocks
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)

    # Multi-Head Attention Layer (to retain most important chunks within the data rather than the whole time series)
    x = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = Dropout(0.3)(x)

    # LSTM
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(x)
    x = LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))(x)

    # Dense layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.4)(x)

    # Output layer
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#model fit

model = create_model()

# Early stopping - we might need it 
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
   # validation_data=(X_val_train, y_val_train),
   # callbacks=[early_stopping],
    verbose=1
)

#Predicting on validation set
# Ensure predictions are class labels (we'll use argmax since we have probabilities distribution)
y_pred_probs = model.predict(X_val)  # probability predictions for the test set
y_pred = np.argmax(y_pred_probs, axis=1)  # convert probabilities to class labels


y_true = y_val

# Ensure y_true contains class labels (same procedure as the predicted labels)
y_true = np.argmax(y_val, axis=1)  # Convert one-hot encoded labels back to integer class labels


report = classification_report(y_true, y_pred, target_names = ['Class 0', 'Class 1', 'Class 2'], output_dict=True)
print("Report:", classification_report(y_true, y_pred))
print("Classification Report:")
for label in ['Class 0', 'Class 1', 'Class 2']:
    print(f"\nMetrics for {label}:")
    print(f"Precision: {report[label]['precision']:.4f}")
    print(f"Recall: {report[label]['recall']:.4f}")
    print(f"F1-score: {report[label]['f1-score']:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Accuracy - this is our main metric for now
accuracy = np.trace(cm) / np.sum(cm)
print(f"\nAccuracy: {accuracy:.4f}")

# Specificity (for each class)
specificity = {}
for i in range(cm.shape[0]):
    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # True Negatives for each class
    fp = np.sum(cm[:, i]) - cm[i, i]  # False Positives for each class
    specificity[i] = tn / (tn + fp) if (tn + fp) != 0 else 0

print("\nSpecificity:")
for i in range(cm.shape[0]):
    print(f"Specificity for Class {i}: {specificity[i]:.4f}")

# Print some more metrics
print(f"\nMacro Average Precision: {report['macro avg']['precision']:.4f}")
print(f"Macro Average Recall: {report['macro avg']['recall']:.4f}")
print(f"Macro Average F1-score: {report['macro avg']['f1-score']:.4f}")
