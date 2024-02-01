import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt 

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=["sepal length","sepal width","petal length","petal width","class"])
iris.head()
iris.info()
iris['class'].unique()

X = iris.drop('class', axis=1).values 
Y = iris['class'].values 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# visualize borders (linear SVM)

X2_train = X_train[:,:2]
X2_test = X_test[:,:2]

svc = LinearSVC(dual='auto')
svc.fit(X2_train, Y_train)

acc = svc.score(X2_test, Y_test)
acc_train = svc.score(X2_train, Y_train)

# overfitting
print(f'Accuracy - TEST: {acc} / TRAIN: {acc_train}')

# visualizing borders (graph)
def plot_bounds(X,Y,model=None,classes=None, figsize=(8,6)):
        
    plt.figure(figsize=figsize)
        
    if(model):
        X_train, X_test = X
        Y_train, Y_test = Y
        X = np.vstack([X_train, X_test])
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                             np.arange(y_min, y_max, .02))

        if hasattr(model, "predict_proba"):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=.8)

    plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)
    plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, alpha=0.6)
    
    plt.show()

plot_bounds((X2_train, X2_test), (Y_train, Y_test), svc)

svc = LinearSVC(dual='auto')
svc.fit(X_train, Y_train)

acc = svc.score(X_test, Y_test)
acc_train = svc.score(X_train, Y_train)

print(f'Accuracy FullSet - TEST: {acc} / TRAIN: {acc_train}')