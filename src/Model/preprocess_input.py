import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset():
    """
    load dataset from numpy files we saved in create_database part
    :return: cropped_X , cropped_y
    """
    file_path = 'your_files_path'
    cropped_X = np.load(file_path + '/cropped_X.npy')
    cropped_y = np.load(file_path + 'cropped_y.npy')

    return cropped_X, cropped_y


def train_test_val_split(train_size, test_size, X, y):
    """

    :param train_size: float
    :param test_size: float
    :param X: numpy nd array
    :param y: list
    :return: X_train, y_train, X_valid, y_valid, X_test, y_test numpy nd arrays
    """
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, est_size=test_size / 2)

    print("train x.shape :", X_train.shape, "y.shape :", len(y_train))
    print("validation x.shape :", X_valid.shape, "y.shape :", len(y_valid))
    print("test x.shape :", X_test.shape, "y.shape :", len(y_test))
    num_classes = len(set(y))
    print("number of classes :", num_classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def convert_y(y):
    """
    convert str labels to numeric y

    :param y: our original y list witch contain str labels : ['one' , 'd , 'sin , ...]
     :return: numeric_y list : [1 . 2 , 41 , ...] : decoded y
    """
    y_df = pd.read_csv('your y_df path')
    numeric_y = np.zeros(len(y))
    for i in range(len(y)):
        numeric_y[i] = np.where(y_df['class_names'] == y[i])[0][0]
    return numeric_y


def categorical(y):
    cat_y = to_categorical(y, num_classes=43)
    return cat_y


def reshape_x(x):
    """
    reshape x to our target dims
    :param x:
    :return:
    """
    target_w = 32
    target_h = 32
    reshaped_x = x.reshape(-1, target_w, target_h, 1)
    return reshaped_x


def normalization(x):
    new_x = x / 255
    return new_x


def edit_type(x, y):
    edited_x = x.astype(np.float32)
    edited_y = y.astype(np.float32)
    return edited_x, edited_y


def preprocess(x, y):
    reshaped_x = reshape_x(x)
    numeric_y = convert_y(y)
    cat_y = categorical(numeric_y)
    normal_x = normalization(reshaped_x)
    pre_x, pre_y = edit_type(normal_x, cat_y)
    return pre_x, pre_y
