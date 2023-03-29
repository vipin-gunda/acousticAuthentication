import numpy as np
import json

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

#https://www.kaggle.com/code/vishalkesti/feature-extraction-and-fine-tunning-cnn

# step 1: get CNN to run in the first place (hard saving the features and model)
# step 2: later add a CNN check - if CNN is being called and it was already saved (in models folder), just load that pre-trained model and build features from it
# need to double check on shape of data needed

# todo: need to get validation data for CNN model

# ADJUSTABLE PARAMS: PARTICIPANT AND EXPRESSION NUMBER
participant_number = 1
expression_index = 0

# PART 1: PARSE CORRESPONDING DATA
testing_data_path = 'data/testing_data_p'+str(participant_number)+"_e"+str(expression_index)+".txt"
training_labels_path = 'data/training_labels_p'+str(participant_number)+"_e"+str(expression_index)+".txt"
training_data_path = 'data/training_data_p'+str(participant_number)+"_e"+str(expression_index)+".txt"
testing_labels_path = 'data/testing_labels_p'+str(participant_number)+"_e"+str(expression_index)+".txt"

with open(training_data_path) as f:
    content = f.read()
    if content:
        training_data = np.array(json.loads(content)).reshape(-1, 1200, 209, 1)

with open(training_labels_path) as f:
    content = f.read()
    if content:
        training_labels = to_categorical(np.array(json.loads(content)), num_classes=2)

with open(testing_data_path) as f:
    content = f.read()
    if content:
        testing_data = np.array(json.loads(content)).reshape(-1, 1200, 209, 1)
        
with open(testing_labels_path) as f:
    content = f.read()
    if content:
        testing_labels = to_categorical(np.array(json.loads(content)), num_classes=2)
        
# DATA SHAPES
print(training_data.shape)
print(training_labels.shape)
print(testing_data.shape)
print(testing_labels.shape)

# PART 2: DEFINE CNN 
input_shape = (None, 1200, 209, 1)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1200, 209, 1), data_format="channels_last"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# PART 3: TRAIN CNN
# Need to get training_labels and testing_labels
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(training_data, training_labels, epochs=10, validation_data=(testing_data, testing_labels))

# PART 4: GET FEATURES AND WRITE THEM TO EXTRACTED_FEATURES FOLDER
# Get features for both training and testing

# PART 5: SAVE CNN MODEL

# possible issues
# is it because removing last layer of cnn? so model not able to compare output against actual layer
# potentionally have to use tensors instead of numpy arrays
