import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.metrics import accuracy_score
import random
import json

# ADJUSTABLE PARAMS: PARTICIPANT AND EXPRESSION NUMBER
participant_number = 1
expression_index = 0

# PART 1: PARSE CORRESPONDING DATA
# IMPORTANT FOR DEV: Reading from raw compiled data currently
# Later, need to get the CNN feature data (just changing the path)
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

#print(training_data) #set of training data
#print(testing_data) #set of testing data
#print(testing_answers) #set of answers (1 or 0)
#print(testing_expressions) #set of expressions (0...8)

# PART 2: TRAIN SVM ON TRAINING DATA
authen = svm.OneClassSVM()
authen.fit(training_data)

# PART 3: TEST SVM ON TESTING DATA
predictions = authen.predict(testing_data)
#score = accuracy_score(testing_answers, predictions)
#print(score)
print(predictions)