import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file = 'dataset/transfusion.data'

'''
R (Recency - months since last donation),
F (Frequency - total number of donation),
M (Monetary - total blood donated in c.c.),
T (Time - months since first donation), and
a binary variable representing whether he/she donated blood in March 2007 (1 
stand for donating blood; 0 stands for not donating blood).
'''

def load_data():
    with open(file, 'r') as f:
        names = f.readline().strip('\n')

    dataset = pd.read_table(file, names=names.split(','), skiprows=1, delimiter=',')

    return dataset


def split_data(data, labels=-1, test_size=0.50):
    array = data.values
    X = array[:, :labels]
    X = scale_data(X)
    Y = array[:, labels]
    Y = Y.astype('int')

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=test_size)
    # X_predict, X_fit, Y_predict, Y_fit = train_test_split(X_validation, Y_validation, test_size=0.5)

    # return X_train, X_fit, X_predict, Y_train, Y_fit, Y_predict
    return X_train, X_validation, Y_train, Y_validation

def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data

def plot_dataset(data):
    figure, axis = plt.subplots(2, 2)
    axis[0, 0].hist(data['Recency (months)'])
    axis[0, 0].set_title('Recency (months)')
    axis[0, 1].hist(data['Frequency (times)'])
    axis[0, 1].set_title('Frequency (times)')
    axis[1, 0].hist(data['Monetary (c.c. blood)'])
    axis[1, 0].set_title('Monetary (c.c. blood)')
    axis[1, 1].hist(data['Time (months)'])
    axis[1, 1].set_title('Time (months)')
    plt.show()

# if __name__ == '__main__':
#     data = load_data()
#     print(data)
#     print(data['blood donated in March 2007'].value_counts())
#     X_train, X_test, Y_train, Y_test = split_data(data, test_size=0.25)
#     print(np.count_nonzero(Y_train == 0))
#     print(np.count_nonzero(Y_train == 1))
#     print(np.count_nonzero(Y_test == 0))
#     print(np.count_nonzero(Y_test == 1))
#     X_train, X_fit, X_predict, Y_train, Y_fit, Y_predict = split_data(data)
#     plot_dataset(data)