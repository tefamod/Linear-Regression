import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import scipy.optimize as optimization
from scipy.linalg import lstsq
from scipy.optimize import least_squares

def Accuracy(test_y,predict_y):
    sum = np.sum(np.abs(np.subtract(test_y,predict_y)/test_y))
    return (1-(sum/len(predict_y)))*100



def loss(X, y, weights, m):
    predicted = X.dot(weights)
    residual = np.subtract(predicted, y)
    sq_residual = np.square(residual)
    Loss = (1 / (2 * m)) * np.sum(sq_residual)
    return Loss


def Gradient_Descent(X, y, weights, learning_rate, iterations, m):
    #periteration = np.zeros(iterations)
    # Stochastic Gradient Descent: This is a type of gradient descent which processes 1 training example per iteration.
    for i in range(iterations):
        predicted = X.dot(weights)
        residual = np.subtract(predicted, y)
        delta = (learning_rate / m) * X.transpose().dot(residual)
        weights = weights - delta
        #periteration[i] = loss(X, y, weights, m)
    return weights#, periteration


if __name__ == '__main__':

    p=pd.read_csv("USA_Housing.csv")
    #p.info()
    data = np.matrix
    data[0]=p.head(1)
    data[1] = p.head(2)

    #data += p.head(2)
    print(data)
    train_set,test_set=train_test_split(p,test_size=0.33, random_state=42)
    #print(train_set)
    #print("test\n",test_set)
    label = []
    weights = np.zeros(3)
    iterations = 50000
    learning_rate = 0.15

    #standarized_df = p.copy()
    #    standarized_df.X1 = preprocessing.normalize([standarized_df.X1])[0]
    #standarized_df.X2 = preprocessing.normalize([standarized_df.X2])[0]
    #standarized_df.X3 = preprocessing.normalize([standarized_df.X3])[0]
    #print(standarized_df)

    #x_train, x_test, y_train, y_test = train_test_split(standarized_df.iloc[:, :-1], standarized_df.Label,
                                                   #     test_size=0.33, random_state=42)
    #m = len(x_train)
    #print('x_train:', x_train)
    #print('x_test:', x_test)
    #print('y_train:', y_train)
    #print('y_test:', y_test)
    #res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
    #weights = Gradient_Descent(x_train, y_train, weights, learning_rate, iterations, m)
    #print(weights)
    #prediction = x_test.dot(weights)
    #print('y_predict:', prediction)
    #l = loss(x_test, y_test, weights,m)
   # print("loss cost: ", l)
    #print("---------------------------\nweights:")
    #print(weights)
    #print("---------------------------\nAccuracy =")
    #print(Accuracy(y_test,prediction) + "%")