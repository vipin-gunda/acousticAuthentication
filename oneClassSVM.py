import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

#participant_path contains the location of all the sessions for participant 1.
participant_path = '/data/shawn/authen_acoustic_2023sp/smart_eyewear_user_study/glasses_P1_sitting'

sync1 = participant_path + "/wav_csv_sync_1.txt"
sync2 = participant_path + "/wav_csv_sync_2.txt"

group_one_count = 0
with open(sync1) as f1:
    lines = f1.readlines()
    ta1 = float(lines.pop(0))
    tv1 = float(lines.pop(0).split(",").pop(0))
    group_one_count += 1
    for line in lines:
        print(line)
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

# TO FEED INTO SVM
# Set correct expression you're trying to detect here
expression_index = 0
training_data = []
testing_data = []

def get_npy_frame(index: float, session: int):
    # TODO: Return npy frame from correct file
    pass

# for every other expression (other than the chosen expression index)
    # get random session number between 1-10, get random npy index
    # add to data

# go to the right expression to get the correct 2d array
# go through each session and with each given npy index
    # for first 10 sessions, index into the correct npy file with it (first 6 sessions are in first file, group 2 in second)
        # for x: processing (check w/ ke)
        # get full y
    # concat that frame to the data
    # for last two sessions, do same indexing but add into testing data

# feed all that data into the svm

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1],
                 c="blueviolet", s=s, edgecolors="k")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                c="gold", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend(
    [a.collections[0], b1, b2, c],
    [
        "learned frontier",
        "training observations",
        "new regular observations",
        "new abnormal observations",
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers)
)
plt.savefig("plot.png")
