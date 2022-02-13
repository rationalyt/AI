import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from statistics import mean

data = pd.read_csv(".\data.csv")
kf = KFold(n_splits=12)
model = KNeighborsClassifier(n_neighbors=1,metric = 'manhattan')

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]
acc_score = []

for train_index, test_index in kf.split(x):
    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred_values = model.predict(X_test)
    acc = accuracy_score(pred_values, y_test)
    acc_score.append(acc)
    error = 0
print("\nThe accuracy for 1NN:")
print(acc_score)
print("\nThe error % for 1NN:")
print((1-mean(acc_score))*100)


#------------------------ b ---------------------#

for n in [3,10]:
 dist = []
 index = []
 for i in range(0,len(data)):
     dist.append(abs(data.iloc[n-1,1]-data.iloc[i,1])+abs(data.iloc[n-1,2]-data.iloc[i,2]))
     index.append(i+1)
 z = list(zip(index,dist))
 k = sorted(z,key = lambda m: m[1])
 print("\nThe 3 nearest neighbor IDs for id={0} are:".format(n))
 for j in range(1,4):
     print(k[j][0])



#--------------------- c ------------------------#

train_index,test_index = [],[]
model = KNeighborsClassifier(n_neighbors=3,metric = 'manhattan')
error_score = []
acc_score = []
for i in range(0,3):
    acc = []
    train_index = []
    test_index = []
    for j in range(0,len(data)):
        if data.iloc[j,0]%3 == i:
            test_index.append(data.iloc[j,0]-1)
        else:
            train_index.append(data.iloc[j,0]-1)
    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred_values = model.predict(X_test)
    acc = accuracy_score(pred_values, y_test)
    acc_score.append(acc)
print("\nThe accuracy for 3NN:")
print(acc_score)
print("\nThe error % for 3NN:")
print((1 - mean(acc_score))*100)
