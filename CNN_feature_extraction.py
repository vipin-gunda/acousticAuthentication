import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# inspo from this link: https://www.tensorflow.org/tutorials/images/cnn

# 1. do some data cleaning (put it in correct form for input)
# split training, validation, testing datasets
# paper used spectrogram of the segmented signal as input

# 2. create all cnn layers
# paper has a list of 14 layers
# just do the most basic CNN for now
# remember to remove the last layer! only get new representation, not result

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# do not include dense layers, no need to classify
# might need to flatten last output?

# 3. pass all training dataset into cnn to train (compile and train model)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# now, can use this model to create another representation to pass into SVM
# that will presumably work better

# 4. test model here
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)