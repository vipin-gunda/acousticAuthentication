import numpy as np
import random
import json
import os

# ADJUSTABLE: PARTICIPANT NUMBER + EXPRESSION INDEX
participant_number = 1
expression_index = 0
#participant_path contains the location of all the sessions for the participant in question.
participant_path = '/data/shawn/authen_acoustic_2023sp/smart_eyewear_user_study/glasses_P'+ str(participant_number) +'_sitting'

# PART 1: ORGANIZE RAW DATA INTO EXPRESSIONS + SESSIONS HASHMAP
sync1 = participant_path + "/wav_csv_sync_1.txt"
sync2 = participant_path + "/wav_csv_sync_2.txt"

group_one_count = 0
with open(sync1) as f1:
   lines = f1.readlines()
   ta1 = float(lines.pop(0))
   tv1 = float(lines.pop(0).split(",").pop(0))
   group_one_count += 1
   for line in lines:
       group_one_count += 1

group_two_count = 0
with open(sync2) as f2:
   lines = f2.readlines()
   ta2 = float(lines.pop(0))
   tv2 = float(lines.pop(0).split(",").pop(0))
   group_two_count += 1
   for line in lines:
       group_two_count += 1

expression_map = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}

total_sessions = group_one_count + group_two_count
for i in range(0, 8 + 1):
   for j in range(1, total_sessions + 1):
       expression_map[i].append([])

for i in range(1, group_one_count + 1):
   txt_path = participant_path + "/session_" + str(i) + "/facial_expr_timestamp.txt"
   with open(txt_path) as txt:
       lines = txt.readlines()
       for line in lines:
           info_list = line.split(", ")
           t = float(info_list[0])
           exp = int(info_list[2])
           npy_index = int(((t - tv1) * 50000 + ta1) / 600)
           expression_map[exp][i-1].append(npy_index)

for i in range(group_one_count + 1, total_sessions + 1):
   txt_path = participant_path + "/session_" + str(i) + "/facial_expr_timestamp.txt"
   with open(txt_path) as txt:
       lines = txt.readlines()
       for line in lines:
           info_list = line.split(", ")
           t = float(info_list[0])
           exp = int(info_list[2])
           npy_index = int(((t - tv2) * 50000 + ta2) / 600)
           expression_map[exp][i-1].append(npy_index)

# PART 2: READING HASHMAP TO CREATE NPY TRAINING DATA
training_data = []
training_labels = []
testing_data = []
testing_labels = []

npy_file_path = '/data/smart_eyewear/user_study/glasses_P' + str(participant_number) + '_sitting/'
npy_1 = np.load(npy_file_path + 'facial_expression_1_fmcw_diff_CIR.npy')
npy_2 = np.load(npy_file_path + 'facial_expression_2_fmcw_diff_CIR.npy')
duration = int(2.5 * 50000 / 600)

def get_npy_frame(ind: int, session: int):
   shift_seconds = random.uniform(0, 0.5) # randomly generation a float between 0 to 0.5, shift in seconds
   shift_scaled = int(shift_seconds * 50000/600)
   if session <= group_one_count:
       return npy_1[:, ind + shift_scaled:ind + duration + shift_scaled + 1]
   else:
       return npy_2[:, ind + shift_scaled:ind + duration + shift_scaled + 1]

# Compiling Training Data
for exp in range(0, 8 + 1):
   if exp != expression_index:
       random_session_number = random.randint(1, total_sessions-2)
       session_length = len(expression_map[exp][random_session_number-1])
       random_npy_index = expression_map[exp][random_session_number-1][random.randint(0, session_length-1)]
       training_data.append(get_npy_frame(random_npy_index, random_session_number))
       training_labels.append(0)
   else:
       for session_number in range(1, total_sessions - 2 + 1):
               for npy_index in expression_map[exp][session_number - 1]:
                        training_labels.append(1)
                        training_data.append(get_npy_frame(npy_index, session_number))

# Compiling Testing Data
for exp in range(0, 8 + 1):
   if exp != expression_index:
       random_session_number = random.randint(total_sessions - 2 + 1, total_sessions)
       session_length = len(expression_map[exp][random_session_number-1])
       random_npy_index = expression_map[exp][random_session_number-1][random.randint(0, session_length-1)]
       testing_labels.append(0)
       testing_data.append(get_npy_frame(random_npy_index, random_session_number))
   else:
       for session_number in range(total_sessions - 2 + 1, total_sessions + 1):
               for npy_index in expression_map[exp][session_number - 1]:
                       testing_labels.append(1)
                       testing_data.append(get_npy_frame(npy_index, session_number))

# PART 3: WRITING TRAINING AND TESTING DATA TO DATA FOLDER
training_data = np.array(training_data)
training_labels = np.array(training_labels)
testing_data = np.array(testing_data)
testing_labels = np.array(testing_labels)

if not os.path.exists('data'):
   os.makedirs('data') 
   
with open("data/training_data_p" + str(participant_number) + "_e" + str(expression_index) + ".txt", "w") as txt:
    json.dump(training_data.tolist(), txt)
with open("data/training_labels_p" + str(participant_number) + "_e" + str(expression_index) + ".txt", "w") as txt:
    json.dump(training_labels.tolist(), txt)
with open("data/testing_data_p" + str(participant_number) +"_e" + str(expression_index) + ".txt", "w") as txt:
    json.dump(testing_data.tolist(), txt)
with open("data/testing_labels_p" + str(participant_number) +"_e" + str(expression_index) + ".txt", "w") as txt:
    json.dump(testing_labels.tolist(), txt)