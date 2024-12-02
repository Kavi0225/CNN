# Import libraries
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Define file path and label
file_path = r"C:\Users\demon\matfile\MAT\ECGPCG0001.mat"
intensity_label = 0  # Replace with the actual label if needed

# Load and preprocess data
sample_length = 10000  # Standard length for ECG and PCG signals
input_shape = (2, sample_length)  # Define a consistent shape for each fused image

# Load data from the file
data_content = sio.loadmat(file_path)
ecg_signal = data_content.get('ECG')
pcg_signal = data_content.get('PCG')

# Check if both signals are present and have the required length
if ecg_signal is not None and pcg_signal is not None and \
        len(ecg_signal) >= sample_length and len(pcg_signal) >= sample_length:
    # Resize signals and create 2D image
    fused_image = np.vstack((ecg_signal[:sample_length], pcg_signal[:sample_length]))
    data = np.array([fused_image])
    labels = np.array([intensity_label])
else:
    raise ValueError("Insufficient data length or missing ECG/PCG signals.")

# Normalize data
data = data / np.max(data)  # Normalize data

# Convert labels to categorical format
labels = to_categorical(labels)

# Duplicate the single data sample to create a minimal dataset
data = np.concatenate([data, data], axis=0)
labels = np.concatenate([labels, labels], axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape data for CNN input
X_train = X_train.reshape(-1, 2, sample_length, 1)
X_test = X_test.reshape(-1, 2, sample_length, 1)

# Define CNN model
model = Sequential([
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(2, sample_length, 1)),
    MaxPooling2D((3, 3), strides=2),
    Conv2D(256, (5, 5), activation='relu'),
    MaxPooling2D((3, 3), strides=2),
    Conv2D(384, (3, 3), activation='relu'),
    Conv2D(384, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((3, 3), strides=2),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=1, validation_data=(X_test, y_test))

# Evaluate and print accuracy
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_true, y_pred_classes)
print("Test Accuracy:", accuracy)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)
