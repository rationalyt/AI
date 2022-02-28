import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score
import numpy as np
#Task 1
df = pd.read_csv("gthanu.csv")
p = list(df.columns)
data = []
df.columns = ["X1", "X2", "X3", "X4", "X5", "Y"]
data.insert(0, {'X1': float(p[0]), 'X2': float(p[1]), 'X3': float(p[2]), 'X4' : float(p[3]), 'X5' : float(p[4]), 'Y': float(p[5])})
df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
"""print(df.mean())
print("\n")
print(df.var())
df.hist(bins = 20)
plt.show()
print(df.corr())"""


def show(model,X1,Y):
    plt.plot(X1,model.predict(X1),color='green')
    plt.title("Regression Model")
    plt.xlabel("X1")
    plt.ylabel("Y")
    plt.show()
    print("Slope: {0}".format(model.coef_))
    print("Intercept: {0}".format(model.intercept_))
    ypredict = model.predict(X1)
    residual = ypredict - list(df['Y'])
    print ("Coefficient of determination(R**2) :",r2_score(Y,ypredict))
    print ("MSE(sigma**2): ",mean_squared_error(Y,ypredict))

    plt.hist(residual)
    plt.title("Historgram Residual")
    plt.show()

    plt.scatter(list(df['Y']),residual)
    plt.title("Scatter Plot")
    plt.show()

    scipy.stats.probplot(residual, dist="norm", plot=plt)
    plt.show()

    print(scipy.stats.chisquare(residual))


#Task 2
"""print(list(df['X1']))
#print(df.loc[:,'X1'])

slope, intercept, r, p, std_err = stats.linregress(list(df['X1']), list(df['Y']))

print(slope)
print(intercept)
print(r)
print(p)
print(std_err)"""


X1,Y = np.array(df['X1']).reshape(-1,1), np.array(list(df['Y']))
model = LinearRegression()
pred = model.fit(X1, Y)
show(model,X1,Y)
print("\n")


#Task 3
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X1)
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, Y)
show(poly_reg_model,poly_features,Y)
print("\n")

X1,Y = np.array(df.iloc[:,0:5]), np.array(list(df['Y']))
model = LinearRegression()
pred = model.fit(X1, Y)
show(model,X1,Y)







