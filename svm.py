import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt



data = pd.read_csv("./svm/svm_2021.csv")
print(data['Class'].value_counts())
X_train, X_test, Y_train, Y_test = train_test_split(data.loc[:, data.columns != 'Class'], data['Class'], stratify=data['Class'], train_size=0.8)
print("\nThe number of Class0 and Class1 samples in training data")
print(Y_train.value_counts())
print("\nThe number of Class0 and Class1 samples in testing data")
print(Y_test.value_counts())

#------------------------ 5c ------------------------------#

C = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]
model_list = []
training_svm = pd.read_csv("./svm/train_data_2021.csv")
testing_svm = pd.read_csv("./svm/test_data_2021.csv")
for c in C:
    model = SVC(C = c, kernel = 'linear')
    model_list.append(model.fit(training_svm.loc[:, data.columns != 'Class'], training_svm['Class']))
sv_count = []
for m in model_list:
    sv_count.append(m.n_support_[0] + m.n_support_[1])
plt.plot(C,sv_count)
plt.xlabel("C")
plt.ylabel("Support Vectors")
plt.scatter(C,sv_count)
plt.show()

C = [0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000]
degree = [1, 2, 3, 4, 5]
coef0 = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10]
gamma = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3]
kernel = ['linear', 'rbf', 'poly', 'sigmoid']

svc = SVC()

parameter_linear = {'kernel': [kernel[0]], 'C':C}
clf = GridSearchCV(svc, parameter_linear, cv = 5, refit = 'accuracy')
clf.fit(training_svm.loc[:, training_svm.columns != 'Class'], training_svm['Class'])
y_linear = clf.predict(testing_svm.loc[:, training_svm.columns != 'Class'])

parameter_rbf = {'kernel': [kernel[1]], 'C':C, 'gamma':gamma}
clf_rbf = GridSearchCV(svc, parameter_rbf, cv = 5, refit = 'accuracy')
clf_rbf.fit(training_svm.loc[:, training_svm.columns != 'Class'], training_svm['Class'])
y_rbf = clf_rbf.predict(testing_svm.loc[:, testing_svm.columns != 'Class'])

parameter_poly = {'kernel': [kernel[2]], 'C': C, 'degree': degree, 'coef0': coef0}
clf_poly = GridSearchCV(svc, parameter_poly, cv = 5, refit = 'accuracy')
clf_poly.fit(training_svm.loc[:, training_svm.columns != 'Class'], training_svm['Class'])
y_poly = clf_poly.predict(testing_svm.loc[:, testing_svm.columns != 'Class'])


parameter_sigmoid = {'kernel': [kernel[3]], 'C':C, 'gamma': gamma, 'coef0': coef0}
clf_sigmoid = GridSearchCV(svc, parameter_sigmoid, cv = 5, refit = 'accuracy')
clf_sigmoid.fit(training_svm.loc[:, training_svm.columns != 'Class'], training_svm['Class'])
y_sigmoid = clf_sigmoid.predict(testing_svm.loc[:, testing_svm.columns != 'Class'])

print("\nMetrics for Linear")
print(f"Accuracy: {metrics.accuracy_score(testing_svm['Class'],y_linear)}")
print(f"F1score: {metrics.f1_score(testing_svm['Class'],y_linear)}")
print(f"Precision: {metrics.precision_score(testing_svm['Class'],y_linear)}")
print(f"Recall: {metrics.recall_score(testing_svm['Class'],y_linear)}")
print(f"C: {clf.best_estimator_.C}")
print(f"deg: {clf.best_estimator_.degree}")
print(f"C0: {clf.best_estimator_.coef0}")
print(f"gamma: --")

print("\nMetrics for Poly:")
print(f"Accuracy: {metrics.accuracy_score(testing_svm['Class'],y_poly)}")
print(f"F1score: {metrics.f1_score(testing_svm['Class'],y_poly)}")
print(f"Precision: {metrics.precision_score(testing_svm['Class'],y_poly)}")
print(f"Recall: {metrics.recall_score(testing_svm['Class'],y_poly)}")
print(f"C: {clf_poly.best_estimator_.C}")
print(f"deg: {clf_poly.best_estimator_.degree}")
print(f"C0: {clf_poly.best_estimator_.coef0}")
print(f"gamma: --")

print("\nMetrics for RBF:")
print(f"Accuracy: {metrics.accuracy_score(testing_svm['Class'],y_rbf)}")
print(f"F1score: {metrics.f1_score(testing_svm['Class'],y_rbf)}")
print(f"Precision: {metrics.precision_score(testing_svm['Class'],y_rbf)}")
print(f"Recall: {metrics.recall_score(testing_svm['Class'],y_rbf)}")
print(f"C: {clf_rbf.best_estimator_.C}")
print(f"deg: {clf_rbf.best_estimator_.degree}")
print(f"C0: {clf_rbf.best_estimator_.coef0}")
print(f"gamma: {clf_rbf.best_estimator_.gamma}")

print("\nMetrics for Sigmoid:")
print(f"Accuracy: {metrics.accuracy_score(testing_svm['Class'],y_sigmoid)}")
print(f"F1score: {metrics.f1_score(testing_svm['Class'],y_sigmoid)}")
print(f"Precision: {metrics.precision_score(testing_svm['Class'],y_sigmoid)}")
print(f"Recall: {metrics.recall_score(testing_svm['Class'],y_sigmoid)}")
print(f"C: {clf_sigmoid.best_estimator_.C}")
print(f"deg: {clf_sigmoid.best_estimator_.degree}")
print(f"C0: {clf_sigmoid.best_estimator_.coef0}")
print(f"gamma: {clf_sigmoid.best_estimator_.gamma}")
