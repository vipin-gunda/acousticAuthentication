import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models
from keras.models import Model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# https://www.kaggle.com/code/vishalkesti/feature-extraction-and-fine-tunning-cnn

# step 1: get CNN to run in the first place (hard saving the features and model)
# step 2: later add a CNN check - if CNN is being called and it was already saved (in models folder), just load that pre-trained model and build features from it
# need to double check on shape of data needed

# ADJUSTABLE PARAMS: PARTICIPANT AND EXPRESSION NUMBER
participant_number = 2
session_date = "0417"
# expression_index = 0

# PART 1: PARSE CORRESPONDING DATA
testing_data_path = 'data/' + session_date + '/testing_data' + ".txt"
testing_labels_path = 'data/' + session_date + '/testing_labels_p' + \
    str(participant_number) + ".txt"
training_data_path = 'data/' + session_date + '/training_data' + ".txt"
training_labels_path = 'data/' + session_date + '/training_labels_p' + \
    str(participant_number) + ".txt"

with open(training_data_path) as f:
    content = f.read()
    if content:
        training_data = np.array(json.loads(
            content)).reshape(-1, 1200, 2166, 1)

with open(training_labels_path) as f:
    content = f.read()
    if content:
        training_labels = np.array(json.loads(content))

with open(testing_data_path) as f:
    content = f.read()
    if content:
        testing_data = np.array(json.loads(content)).reshape(-1, 1200, 2166, 1)

with open(testing_labels_path) as f:
    content = f.read()
    if content:
        testing_labels = np.array(json.loads(content))

# PART 2: DEFINE CNN
input_shape = (None, 1200, 2166, 1)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=(1200, 2166, 1), data_format="channels_last"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation='sigmoid'))

# PART 3: TRAIN CNN
# Need to get training_labels and testing_labels
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(training_data, training_labels, epochs=10,
                    validation_data=(testing_data, testing_labels), batch_size=3)

# PART 4: GET FEATURES AND WRITE THEM TO EXTRACTED_FEATURES FOLDER
# Remove last Dense layer and just return the output
model2 = Model(model.input, model.layers[-2].output)
model2.summary()

# PART 5: SAVE CNN MODEL
# https://www.tensorflow.org/guide/keras/save_and_serialize
model2.save('models/' + session_date + "/p_" + str(participant_number))

# TODO: LOAD MODEL AND RUN ON DATASET FOR EXTRACTED FEATURES
