import numpy as np
import random
import json
import os

# ADJUSTABLE: PARTICIPANT NUMBER
participant_number = 1
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
# ADJUSTABLE: Set correct expression you're trying to detect here
expression_index = 0
training_data = []
testing_data = []
testing_answers = []
testing_expressions = []

npy_file_path = '/data/smart_eyewear/user_study/glasses_P' + str(participant_number) + '_sitting/'
npy_1 = np.load(npy_file_path + 'facial_expression_1_fmcw_diff_CIR.npy')
npy_2 = np.load(npy_file_path + 'facial_expression_2_fmcw_diff_CIR.npy')
duration = int(2.5 * 50000 / 600)

# session #1-12
def get_npy_frame(ind: int, session: int):
   shift_seconds = random.uniform(0, 0.5) # randomly generation a float between 0 to 0.5, shift in seconds
   shift_scaled = int(shift_seconds * 50000/600)
   if session <= group_one_count:
       return npy_1[:, ind + shift_scaled:ind + duration + shift_scaled + 1].flatten("F") #CHECK FLATTENING
   else:
       return npy_2[:, ind + shift_scaled:ind + duration + shift_scaled + 1].flatten("F") #CHECK FLATTENING

for exp in range(0, 8 + 1):
   if exp != expression_index:
       random_session_number = random.randint(1, total_sessions-2)
       session_length = len(expression_map[exp][random_session_number-1])
       random_npy_index = expression_map[exp][random_session_number-1][random.randint(0, session_length-1)]
       #np.append(training_data, get_npy_frame(random_npy_index, random_session_number))
       training_data.append(get_npy_frame(random_npy_index, random_session_number))
   else:
       for session_number in range(1, total_sessions - 2 + 1):
               for npy_index in expression_map[exp][session_number - 1]:
                       #np.append(training_data,get_npy_frame(npy_index, session_number))
                       training_data.append(get_npy_frame(npy_index, session_number))

for exp in range(0, 8 + 1):
   if exp != expression_index:
       random_session_number = random.randint(total_sessions - 2 + 1, total_sessions)
       session_length = len(expression_map[exp][random_session_number-1])
       random_npy_index = expression_map[exp][random_session_number-1][random.randint(0, session_length-1)]
       #np.append(testing_answers, -1)
       testing_answers.append(-1)
       #np.append(testing_expressions, exp)
       testing_expressions.append(exp)
       #np.append(testing_data, get_npy_frame(random_npy_index, random_session_number))
       testing_data.append(get_npy_frame(random_npy_index, random_session_number))
   else:
       for session_number in range(total_sessions - 2 + 1, total_sessions + 1):
               for npy_index in expression_map[exp][session_number - 1]:
                       #np.append(testing_answers,1)
                       testing_answers.append(1)
                       #np.append(testing_expressions, exp)
                       testing_expressions.append(exp)
                       #np.append(testing_data, get_npy_frame(npy_index, session_number))
                       testing_data.append(get_npy_frame(npy_index, session_number))

# PART 3: WRITING TRAINING AND TESTING DATA TO DATA FOLDER
training_data = np.array(training_data)
testing_data = np.array(testing_data)

current_path = os.getcwd()
data_path = current_path + "/data"

if not os.path.exists(data_path):
   os.makedirs('data') 
   
with open(data_path + "/training_data_" + str(expression_index) + ".txt", "w") as txt:
 json.dump(training_data.tolist(), txt)


with open(data_path + "/testing_data_" + str(expression_index) + ".txt", "w") as txt:
 json.dump(testing_data.tolist(), txt)