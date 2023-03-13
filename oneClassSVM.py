import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.metrics import accuracy_score
import random

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

print("Group 1 Session Count: " + str(group_one_count))

group_two_count = 0
with open(sync2) as f2:
    lines = f2.readlines()
    ta2 = float(lines.pop(0))
    tv2 = float(lines.pop(0).split(",").pop(0))
    group_two_count += 1
    for line in lines:
        group_two_count += 1

print("Group 2 Session Count: " + str(group_two_count))

expression_map = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
print("Initial Expression Map")
print(expression_map)

total_sessions = group_one_count + group_two_count
print("Total Number of Sessions: " + str(total_sessions))
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

print("Final Expression Map")
print(expression_map)

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
duration = int(2.2333 * 50000 / 600)

# session #1-12
def get_npy_frame(ind: int, session: int):
    if session <= group_one_count:
        return npy_1[:, ind:ind + duration + 1].flatten("F") #CHECK FLATTENING
    else:
        return npy_2[:, ind:ind + duration + 1].flatten("F") #CHECK FLATTENING

for exp in range(0, 8 + 1):
    if exp != expression_index:
        random_session_number = random.randint(1, total_sessions-2)
        session_length = len(expression_map[exp][random_session_number-1])
        random_npy_index = expression_map[exp][random_session_number-1][random.randint(0, session_length-1)]
        training_data.append(get_npy_frame(random_npy_index, random_session_number))
    else:
        for session_number in range(1, total_sessions - 2 + 1):
                for npy_index in expression_map[exp][session_number - 1]:
                        training_data.append(get_npy_frame(npy_index, session_number))

for exp in range(0, 8 + 1):
    if exp != expression_index:
        random_session_number = random.randint(total_sessions - 2 + 1, total_sessions)
        session_length = len(expression_map[exp][random_session_number-1])
        random_npy_index = expression_map[exp][random_session_number-1][random.randint(0, session_length-1)]
        testing_answers.append(-1)
        testing_expressions.append(exp)
        testing_data.append(get_npy_frame(random_npy_index, random_session_number))
    else:
        for session_number in range(total_sessions - 2 + 1, total_sessions + 1):
                for npy_index in expression_map[exp][session_number - 1]:
                        testing_answers.append(1)
                        testing_expressions.append(exp)
                        testing_data.append(get_npy_frame(npy_index, session_number))

print(training_data)
print(testing_data)
print(testing_answers)
print(testing_expressions)

# PART 3: TRAIN SVM ON TRAINING DATA
# train SVM on training_data
# get results w/ testing data
# get accuracy
# visualize SVM

# fit the model
authen = svm.OneClassSVM()
authen.fit(training_data)
predictions = authen.predict(testing_data)
score = accuracy_score(testing_answers, predictions)
print(score)
print(predictions)

# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
# n_error_train = y_pred_train[y_pred_train == -1].size
# n_error_test = y_pred_test[y_pred_test == -1].size
# n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# # plot the line, the points, and the nearest vectors to the plane
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.title("Novelty Detection")
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

# s = 40
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1],
#                  c="blueviolet", s=s, edgecolors="k")
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
#                 c="gold", s=s, edgecolors="k")
# plt.axis("tight")
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend(
#     [a.collections[0], b1, b2, c],
#     [
#         "learned frontier",
#         "training observations",
#         "new regular observations",
#         "new abnormal observations",
#     ],
#     loc="upper left",
#     prop=matplotlib.font_manager.FontProperties(size=11),
# )
# plt.xlabel(
#     "error train: %d/200 ; errors novel regular: %d/40 ; errors novel abnormal: %d/40"
#     % (n_error_train, n_error_test, n_error_outliers)
# )
# #plt.savefig("plot.png")
