#
# author: Andrea Milanta
#

#                                                       #
# --------------- IMPORTS ----------------------------- #
# 
# generic
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# splitting and crossvalidation
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import StratifiedKFold

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# classifiers
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB

# monkeys
import monkeyconstants as mc 


#                                                       #
# --------------- LOAD DATASET ------------------------ #
#
csvpath = mc.DATA + "main4h.csv"
df = pd.read_csv(csvpath, sep=',')

# drop non-meaningful features
df.drop("id", axis=1, inplace=True)
df.drop("Length", axis=1, inplace=True)

#                                                       #
# --------------- FEATURE HANDLING -------------------- #
#
# transform target from (1,2) to (1,0)
df["y"] = df["y"].apply(lambda x: 0 if x == 2 else 1)
print("\ndataset balance before dropping null values:")
print(df["y"].value_counts())

# drop null values
lnpre = len(df)
df.dropna(inplace=True)
print("\nThere are %d rows with null values" % (lnpre - len(df)))
print("\ndataset balance after dropping null values:")
print(df["y"].value_counts())

# balancing
print("\ndataset percentage balance after dropping null values:")
print(df["y"].value_counts() / len(df))

# drop dependent features
# df.drop("Subnumber", axis=1, inplace=True)
# df.drop("StraightnessSD", axis=1, inplace=True)
# df.drop("SpeedAvg", axis=1, inplace=True)
df.drop("SpeedSD", axis=1, inplace=True)

# correlation matrix
crmtx = df.corr();

#                                                       #
#------------ CLASSIFICATION X-VALIDATION --------------#
#
#setup train
x_train = df.drop("y", axis=1)
y_train = df["y"] 

# Logistic Regression - CROSS VALIDATION
lr_clf = LogisticRegression(solver='lbfgs')
lr_pred = cross_val_predict(lr_clf, x_train, y_train, cv=2)
lr_acc = accuracy_score(y_train, lr_pred)
lr_prec = precision_score(y_train, lr_pred)
lr_recall = recall_score(y_train, lr_pred)
lr_f1 = f1_score(y_train, lr_pred)

# Decision Tree
dt_clf = tree.DecisionTreeClassifier()
dt_pred = cross_val_predict(dt_clf, x_train, y_train, cv=10)
dt_acc = accuracy_score(y_train, dt_pred)
dt_prec = precision_score(y_train, dt_pred)
dt_recall = recall_score(y_train, dt_pred)
dt_f1 = f1_score(y_train, dt_pred)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=18)
rf_pred = cross_val_predict(rf_clf, x_train, y_train, cv=10)
rf_acc = accuracy_score(y_train, rf_pred)
rf_prec = precision_score(y_train, rf_pred)
rf_recall = recall_score(y_train, rf_pred)
rf_f1 = f1_score(y_train, rf_pred)

# feature importance
dt_clf.fit(x_train, y_train)
importances = dt_clf.feature_importances_
feature_names = x_train.columns
indices = np.argsort(importances)[::-1]


#                                                       #
#------------ RESULTS PRESENTATION ---------------------#
#
# Values
print("\nLogistic Regression:\n\taccuracy: %2.2f \n\tprecision: %2.2f\n\trecall: %2.2f\n\tf1: %2.2f" % (lr_acc, lr_prec, lr_recall, lr_f1))
print("\nDecision Tree:\n\taccuracy: %2.2f \n\tprecision: %2.2f\n\trecall: %2.2f\n\tf1: %2.2f" % (dt_acc, dt_prec, dt_recall, dt_f1))
print("\nRandom Forest:\n\taccuracy: %2.2f \n\tprecision: %2.2f\n\trecall: %2.2f\n\tf1: %2.2f" % (rf_acc, rf_prec, rf_recall, rf_f1))
# # Print the feature ranking
# print("Feature ranking:")       
# for f in range(x_train.shape[1]):
#     print("\t%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # correlation matrix
# f, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(crmtx,annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
# plt.subplots_adjust(left=0.15, right=0.99, bottom=0.25, top=1)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=14)

# Plot the feature importances of the forest
def feature_importance_graph(indices, importances, feature_names):
    f, ax = plt.subplots(figsize=(12, 8))
    plt.title("Determining Feature importances \n with DecisionTreeClassifier", fontsize=18)
    plt.subplots_adjust(left=0.15, right=0.94, bottom=0.11, top=0.90)
    plt.barh(range(len(indices)), importances[indices], color='#31B173',  align="center")
    plt.yticks(range(len(indices)), feature_names[indices], rotation='horizontal',fontsize=14)
    plt.ylim([-1, len(indices)])
    # plt.axhline(y=1.85, xmin=0.21, xmax=0.952, color='k', linewidth=3, linestyle='--')
# feature_importance_graph(indices, importances, feature_names)

# # Logistic Regression confusion matrix
lr_conf = confusion_matrix(y_train, lr_pred)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(lr_conf, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Logistic Regression Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.10, top=0.95)
ax.set_yticks(np.arange(lr_conf.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(['View(0)\npredicted', 'Memory(1)\npredicted'], fontsize=16)
ax.set_yticklabels(['View(0)\nreal', 'Memory(1)\nreal'], fontsize=16, rotation=360)

# # Decision Tree confusion matrix
dt_conf = confusion_matrix(y_train, dt_pred)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(dt_conf, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Decision Tree Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.10, top=0.95)
ax.set_yticks(np.arange(dt_conf.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(['View(0)\npredicted', 'Memory(1)\npredicted'], fontsize=16)
ax.set_yticklabels(['View(0)\nreal', 'Memory(1)\nreal'], fontsize=16, rotation=360)

# # Logistic Regression confusion matrix
rf_conf = confusion_matrix(y_train, rf_pred)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(rf_conf, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Random Forest Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.10, top=0.95)
ax.set_yticks(np.arange(rf_conf.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(['View(0)\npredicted', 'Memory(1)\npredicted'], fontsize=16)
ax.set_yticklabels(['View(0)\nreal', 'Memory(1)\nreal'], fontsize=16, rotation=360)

plt.show()
