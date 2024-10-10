# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.utils import to_categorical
#
# (X_train,y_train), (X_test, y_test) = cifar10.load_data()
# # print(np.unique(y_train))
#
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#
# # Normalize pixel
# X_train = X_train/255.0
# X_test = X_test/255.0
# # #
# # model = models.Sequential()
# # model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)))
# # model.add(BatchNormalization())
# # model.add(layers.MaxPooling2D((2,2)))
# # model.add(layers.Dropout(0.25))
# # model.add(layers.Conv2D(64,(3,3),activation='relu'))
# # model.add(BatchNormalization())
# # model.add(layers.MaxPooling2D(2,2))
# # model.add(layers.Dropout(0.25))
# # model.add(layers.Conv2D(64,(3,3),activation='relu'))
# # model.add(BatchNormalization())
# # model.add(layers.MaxPooling2D(2,2))
# # model.add(layers.Dropout(0.25))
# # model.add(layers.Flatten())
# # model.add(layers.Dense(64, activation='relu'))
# # model.add(layers.Dense(10, activation='softmax'))
# #
# # model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# # model.fit(X_train,y_train,epochs=10,batch_size=64,validation_split=0.1)
# #
# # model_result = model.evaluate(X_test, y_test, verbose=0)
# # print('Accuracy of CNN model: %s'%(model_result[1]*100))
# # model.save('cifar10_cnn_model.h5')

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
(X_train,y_train), (X_test, y_test) = cifar10.load_data()
# print(np.unique(y_train))


model = load_model('cifar10_cnn_model.h5')
prediction  = model.predict(X_test)
predict = np.argmax(prediction[6])

print(predict)

import matplotlib.pyplot as plt
plt.imshow(X_test[6])
plt.title(f'prediction;{predict}')
plt.show()