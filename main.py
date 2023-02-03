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

def Random(start, end,number_of_data):
    # label array to save the value of y
    label = []
    # 3d feature array to save the value of x1,x2,x3
    feature = [[], [], []]
    # loop to set n samples of random values
    for i in range(number_of_data):
        x1 = random.randint(start, end)
        feature[0].append(x1)

        x2 = random.randint(start, end)
        feature[1].append(x2)

        x3 = random.randint(start, end)
        feature[2].append(x3)

        y = 5 * x1 + 3 * x2 + 1.5 * x3 + 6
        label.append(y)
    data_set = {'X1': feature[0], 'X2': feature[1], 'X3': feature[2], 'Label': label}
    show = pd.DataFrame(data_set)

    print(show)

    return feature, label, show


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
    data = [[], [], []]
    label = []
    weights = np.zeros(3)
    iterations = 50000
    learning_rate = 0.15
    data, label, sets = Random(0,300,1000)
    standarized_df = sets.copy()
    standarized_df.X1 = preprocessing.normalize([standarized_df.X1])[0]
    standarized_df.X2 = preprocessing.normalize([standarized_df.X2])[0]
    standarized_df.X3 = preprocessing.normalize([standarized_df.X3])[0]
    #print(standarized_df)

    x_train, x_test, y_train, y_test = train_test_split(standarized_df.iloc[:, :-1], standarized_df.Label,
                                                        test_size=0.33, random_state=42)
    m = len(x_train)
    #print('x_train:', x_train)
    #print('x_test:', x_test)
    #print('y_train:', y_train)
    print('y_test:', y_test)
    #res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
    weights = Gradient_Descent(x_train, y_train, weights, learning_rate, iterations, m)
    print(weights)
    prediction = x_test.dot(weights)
    print('y_predict:', prediction)
    l = loss(x_test, y_test, weights,m)
    print("loss cost: ", l)
    print("---------------------------\nweights:")
    print(weights)
    print("---------------------------\nAccuracy =")
    print(Accuracy(y_test,prediction) + "%")
