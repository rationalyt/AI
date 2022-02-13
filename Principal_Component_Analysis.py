import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

predicted = []
accuracy = []
pc=[]


#----------Method for PCA dimensionality reduction and KNN------------#

def PCAnalysis(n,testset,trainset,train_eigen_vector,type=0):
  col = []
  for i in range(0,n):
    col.append("Principal_Component{0}".format(i+1))
  eigen_train_subset = train_eigen_vector[:,0:n]
  principal_test_set = np.dot(eigen_train_subset.transpose(),testset.iloc[:,0:30].transpose()).transpose()
  principal_train_set = np.dot(eigen_train_subset.transpose(),trainset.iloc[:,0:30].transpose()).transpose()
  count = 0
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(principal_train_set,trainset.iloc[:,30])
  predicted = knn.predict(principal_test_set).tolist()
  actual = testset.iloc[:,30].tolist()
  for i in range(0,len(actual)):
    if actual[i] != predicted[i]:
      count=count+1
  accuracy.append(((len(actual)-count)/len(actual))*100)
  pc.append(n)
  if n == 10:
    df = pd.DataFrame(principal_test_set, columns=col)
    df.insert(10,"Actual_Class",actual)
    df.insert(11,"Predicted Class",predicted)
    df.to_csv('{0}_predicted.csv'.format(type))
  print("Accuracy of Component {0}".format(n))
  print(accuracy[-1])


#-------------------Method for list of PCA to be calculated for different n components---------#

def PCA_List(train_eigen_vector,test_set,train_set,type):

 PCAnalysis(2,test_set,train_set,train_eigen_vector)
 PCAnalysis(4,test_set,train_set,train_eigen_vector)
 PCAnalysis(8,test_set,train_set,train_eigen_vector)
 PCAnalysis(10,test_set,train_set,train_eigen_vector,type=type)
 PCAnalysis(20,test_set,train_set,train_eigen_vector)
 PCAnalysis(25,test_set,train_set,train_eigen_vector)
 PCAnalysis(30,test_set,train_set,train_eigen_vector)


def Normalise(set,train_set):
 nset = set
 for i in range(0, len(set.columns)-1):
  minn = train_set.iloc[:, i].min()
  maxx = train_set.iloc[:, i].max()
  diff = maxx - minn
  for j in range(0, len(set)):
   nset.iloc[j, i] = (nset.iloc[j,i] - minn) / diff
 return nset

def Standardise(set,train_set):
 sset = set
 for i in range(0, len(sset.columns)-1):
  mean = train_set.iloc[:, i].mean()
  stddev = np.std(train_set.iloc[:,i])
  for j in range(0, len(set)):
   sset.iloc[j,i] = (sset.iloc[j,i] - mean)/stddev
 return sset


#---------------------------- Load Data------------------------#

train_set = pd.read_csv(".\data\pca_train.csv")
test_set = pd.read_csv(".\data\pca_test.csv")
print("The size of train_set shape:{0}".format(train_set.shape))
print("The size of test_set shape:{0}".format(test_set.shape))
class1,class0 = 0,0

#---------------------- Class 0, Class 1 count -----------------#
for i in range(0,len(train_set)):
  if train_set.loc[i,'Class']==1:
    class1 = class1 + 1
  if train_set.loc[i,'Class']==0:
    class0 = class0 + 1
print("The number of class0 in train_set : {0}".format(class0))
print("The number of class1 in train_set : {0}".format(class1))
print("\n")
class1,class0 = 0,0
for i in range(0,len(test_set)):
  if test_set.loc[i]['Class']==1:
    class1 = class1 + 1
  if test_set.loc[i]['Class']==0:
    class0 = class0 + 1
print("The number of class0 in test_set: {0}".format(class0))
print("The number of class1 in test_set: {0}".format(class1))


#----------------------- Normalisation & Standardisation of Original Dataset -----------------#


normalised_test_set = Normalise(pd.read_csv(".\data\pca_test.csv"),pd.read_csv(".\data\pca_train.csv"))
normalised_train_set = Normalise(pd.read_csv(".\data\pca_train.csv"),pd.read_csv(".\data\pca_train.csv"))
standardised_test_set = Standardise(pd.read_csv(".\data\pca_test.csv"),pd.read_csv(".\data\pca_train.csv"))
standardised_train_set = Standardise(pd.read_csv(".\data\pca_train.csv"),pd.read_csv(".\data\pca_train.csv"))




#----------------------- Covariance Matrix and Eigen Vector of Normalised Training Dataset--------#
list1 = []
for i in range(0,len(normalised_train_set.columns)-1):
  list1.append(np.array(normalised_train_set.iloc[:,i]))
data = np.array(list1)
train_covMatrix = np.cov(data,bias=True)
print("\nShape of New Normalised Training DataSet Covariance Matrix:")
print(np.shape(train_covMatrix))
print("\nThe 5x5 Covariance Matrix of Normalised Training Data Set:")
for i in range(0,5):
   print(train_covMatrix[i,0:5])
w,v = eig(train_covMatrix)
train_eigen_value = []
train_eigen_list=[]
for x,y in sorted(zip(w,v),reverse=True):
    train_eigen_value.append(x)
    train_eigen_list.append(y)
train_eigen_vector = np.array(train_eigen_list)
print("\nThe first five eigen values of Normalised Training Dataset are:")
for i in range(0,5):
  print(train_eigen_value[i])


#-----------------------------Covariance Matrix Graph for Normalised Dataset ----------------#

fig = plt.figure(figsize = (10,5))
size = []
for i in range(0,len(train_eigen_value)):
  size.append(i)
plt.bar(size,train_eigen_value,color = 'black',width = 0.8)
plt.xlabel("Eigen value count")
plt.ylabel("Eigen values")
plt.title("Covariance Matrix Graph for Normalised Dataset")
plt.show()

#-------------------- PCA on Normalised Training Datasets --------------------#
PCA_List(train_eigen_vector,normalised_test_set,normalised_train_set,'Normalise')
plt.bar(pc, accuracy, color='black', width=0.8)
plt.xlabel("PC value count")
plt.ylabel("Accuracy values")
plt.title("KNN Accuracy of Normalised Dataset")
plt.show()
accuracy.clear()
pc.clear()




#---------------Covariance Matrix and Eigen Vector of Standardised Train Dataset--------#

list1.clear()
for i in range(0,len(standardised_train_set.columns)-1):
  list1.append(np.array(standardised_train_set.iloc[:,i]))
data1 = np.array(list1)
train1_covMatrix = np.cov(data1,bias=True)
print("\nShape of New Standardised Training DataSet Covariance Matrix:")
print(np.shape(train1_covMatrix))
print("\nThe 5x5 in the Covariance Matrix of Standardised Dataset are:")
for i in range(0,5):
   print(train1_covMatrix[i,0:5])
w,v = eig(train1_covMatrix)
train_eigen_value1 = []
train_eigen_list1=[]
for x,y in sorted(zip(w,v),reverse=True):
    train_eigen_value1.append(x)
    train_eigen_list1.append(y)
train_eigen_vector = np.array(train_eigen_list1)
print("\nThe first five eigen values of Standardised Training Dataset are:")
for i in range(0,5):
  print(train_eigen_value1[i])


#---------------Covariance Matrix Graph for Standardised Dataset--------#

fig = plt.figure(figsize = (10,5))
size = []
for i in range(0,len(train_eigen_value)):
  size.append(i)
plt.bar(size,train_eigen_value1,color = 'black',width = 0.8)
plt.xlabel("Eigen value count")
plt.ylabel("Eigen values")
plt.title("Covariance Matrix Graph for Standardised Dataset")
plt.show()

#-------------------- PCA on Standardised Training Datasets --------------------#

PCA_List(train_eigen_vector,standardised_test_set,standardised_train_set,'Standardise')
plt.bar(pc, accuracy, color='black', width=0.8)
plt.xlabel("PC value count")
plt.ylabel("Accuracy values")
plt.title("Standardised Accuracy")
plt.show()
