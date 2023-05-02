import numpy as np
import random
import json
import os

# PART 1: READ IN TIMESTAMPS FROM ALL PARTICIPANTS (2D ARRAY OF TUPLES)
npy_inds = []
p_count = 0
while os.path.isfile('timestamps/0417/timestamps_p'+str(p_count)+'.txt'):
    npy_inds += [[]]  # add new row
    with open('timestamps/0417/timestamps_p'+str(p_count)+'.txt') as f1:
        lines = f1.readlines()
        for i in range(0, len(lines), 2):
            # add start and end timestamps as a tuple, with convert to npy index adjustment
            npy_inds[p_count] += [(int(int(lines[i].split(" ")[0])/600),
                                   int(int(lines[i+1].split(" ")[0])/600))]
    p_count += 1

# PART 2: FOR EACH PARTICIPANT, TAKE NPY SNIPPET AND ADD TO TRAINING DATA
train = []
train_participants = []
test = []
test_participants = []


def get_npy_frame(s_ind: int, e_ind: int, p_ind: int, npy: np.array):
    # randomly generation a float between 0 to 0.5, shift in seconds
    shift_seconds = int(random.uniform(0, 0.5)*50000/600)
    buffer = int(1.5*50000/600)
    duration = int(26*50000/600)
    return npy[:, s_ind + buffer + shift_seconds: s_ind + buffer + duration + shift_seconds]


npy_file_path = '/data/vipin/acousticAuthentication/pilot_study/0417_data/'

for p in range(p_count):
    # p+1 just because of how data files are named
    npy_arr = np.load(npy_file_path + 'facial_expression_' +
                      str(p+1) + '_fmcw_diff_CIR.npy')
    for i, (s, e) in enumerate(npy_inds[p]):
        npy_frame = get_npy_frame(s, e, p, npy_arr)
        if i < len(npy_inds[p])*2/3:  # first 2/3 go to training data
            train += [npy_frame]
            train_participants += [p]
        else:  # last 1/3 go into testing data
            test += [npy_frame]
            test_participants += [p]

if not os.path.exists('data'):
    os.makedirs('data')

with open("data/0417/training_data.txt", "w") as txt:
    json.dump(np.array(train).tolist(), txt)
with open("data/0417/testing_data.txt", "w") as txt:
    json.dump(np.array(test).tolist(), txt)

# PART 3: BUILD TRAINING/TESTING LABELS FOR ALL PARTICIPANTS
for p in range(p_count):
    train_labels_p = [1 if i == p else 0 for i in train_participants]
    test_labels_p = [1 if i == p else 0 for i in test_participants]

    with open("data/0417/training_labels_p" + str(p) + ".txt", "w") as txt:
        json.dump(train_labels_p, txt)

    with open("data/0417/testing_labels_p" + str(p) + ".txt", "w") as txt:
        json.dump(test_labels_p, txt)
