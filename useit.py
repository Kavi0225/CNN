import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import biosppy

# Example path to a .mat file
mat_file_path = r"C:\Users\demon\matfile\MAT\ECGPCG0001.mat"

# Load the .mat file
data = scipy.io.loadmat(mat_file_path)

# Inspect the keys to understand the data structure
print(data.keys())

# Assuming 'ECG' and 'PCG' are keys in the .mat file
ecg_signal = data['ECG'].flatten()
pcg_signal = data['PCG'].flatten()

# Sampling frequency (from Data Description)
fs = 8000  # 8 kHz

# Use BioSPPy to process ECG
ecg_out = biosppy.signals.ecg.ecg(signal=ecg_signal, sampling_rate=fs, show=False)

# R-peaks indices
r_peaks = ecg_out['rpeaks']

# Convert indices to times
r_times = r_peaks / fs

print(f"Detected {len(r_peaks)} R-peaks at times (s): {r_times}")

# Prepare data for modeling (assuming we are classifying based on ECG signal)
# For simplicity, let's divide the ECG signal into smaller windows (e.g., 1000 samples per window)
window_size = 1000
X = []

# Divide ECG signal into windows of size 1000
for i in range(0, len(ecg_signal) - window_size, window_size):
    X.append(ecg_signal[i:i + window_size])

X = np.array(X)  # Convert to numpy array

# Ensure X has multiple samples
print("Shape of X:", X.shape)

# Reshape X to fit model input (samples, time_steps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Labels: Creating binary labels for simplicity, one for each sample
y = np.array([1 if i % 2 == 0 else 0 for i in range(len(X))])  # Dummy binary labels

# Convert labels to categorical
y = to_categorical(y, num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),  # 1D Convolution
    MaxPooling1D(2),  # Max Pooling
    Dropout(0.25),  # Dropout for regularization
    Flatten(),  # Flatten the output for fully connected layers
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout for regularization
    Dense(2, activation='softmax')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Make predictions
predictions = model.predict(X_test)

# Show confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
