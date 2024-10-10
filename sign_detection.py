import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model


h5_file_train = h5py.File('Signs_Data_Training.h5', 'r')

X_train = np.array(h5_file_train['train_set_x'])
Y_train = np.array((h5_file_train['train_set_y']))

h5_file_test = h5py.File('Signs_Data_Testing.h5')
#
x_test = np.array(h5_file_test['test_set_x'])
y_test = np.array(h5_file_test['test_set_y'])
# print(y_test)
X_train = X_train/255.0
x_test = x_test/255.0

# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Flatten(),
#
#     layers.Dense(256, activation='relu'),
#     layers.Dense(10, activation='softmax')  # Adjust number of output units based on the number of classes
# ])
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(X_train,Y_train,epochs=10,batch_size=64,validation_split=0.1)
#
#
# model_result = model.evaluate(x_test, y_test, verbose=0)
# print('Accuracy of CNN model: %s'%(model_result[1]*100))
# model.save('sign_detect.h5')

model = load_model('sign_detect.h5')
prediction  = model.predict(x_test)
predict = np.argmax(prediction[85])

print(predict)

import matplotlib.pyplot as plt
plt.imshow(x_test[85])
plt.title(f'prediction;{predict}')
plt.show()