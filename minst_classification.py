import numpy as np
from sklearn.datasets import fetch_mldata


def main():
    minst = fetch_mldata('MNIST original')
    shuffle_index = np.random.permutation(60000)
    x, y = minst['data'], minst['target']
    index_vale = 60000
    x_train, x_test, y_train, y_test = x[:index_vale], x[index_vale:], y[:index_vale], \
                                       y[index_vale:]
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    y_train_5 = (y_train == 5)


if __name__ == "__main__":
    main()
