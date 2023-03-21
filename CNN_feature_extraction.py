import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#https://www.kaggle.com/code/vishalkesti/feature-extraction-and-fine-tunning-cnn

# step 1: get CNN to run in the first place (hard saving the features and model)
# step 2: later add a CNN check - if CNN is being called and it was already saved (in models folder), just load that pre-trained model and build features from it
# need to double check on shape of data needed

# ADJUSTABLE PARAMS: PARTICIPANT AND EXPRESSION NUMBER
participant_number = 1
expression_index = 0

# PART 1: PARSE CORRESPONDING DATA
testing_data_path = 'data/testing_data_p'+str(participant_number)+"_e"+str(expression_index)+".txt"
training_data_path = 'data/training_data_p'+str(participant_number)+"_e"+str(expression_index)+".txt"

with open(training_data_path) as f:
    content = f.read()
    if content:
        training_data = np.array(json.loads(content))

with open(testing_data_path) as f:
    content = f.read()
    if content:
        testing_data = np.array(json.loads(content))

# PART 2: DEFINE CNN 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

# PART 3: TRAIN CNN
# Need to get training_labels and testing_labels
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(training_data, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# PART 4: GET FEATURES AND WRITE THEM TO EXTRACTED_FEATURES FOLDER
# Get features for both training and testing

# PART 5: SAVE CNN MODEL
