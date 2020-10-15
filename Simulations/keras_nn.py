''' Creates a Keras fully connected NN for testing purposes

Author: Simon Geoffroy-Gagnon
Edit: 2020-04-01
'''

import create_datasets
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
import matplotlib.pyplot as plt

image_size = 10 # 28*28
num_classes = 10 # ten unique digits

model = Sequential()

X, y, Xt, yt = create_datasets.MNIST_dataset(N=10, nsamples=20_000)

model.add(Dense(units=10, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=10, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=10, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, batch_size=128, epochs=2500, verbose=True, validation_split=.1)
loss, accuracy  = model.evaluate(Xt, yt, verbose=False)
# print(history['acc'])
print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.savefig('dnn.png')

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')
print(f'{max(history.history["accuracy"])}')
