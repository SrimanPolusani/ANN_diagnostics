# Import Statements
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import random

# --- START: ROBUST SEEDING BLOCK ---
# This is the most important change. Do it right after your imports.
SEED = 1  # Use a single seed value for everything
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # As per the warning log
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# --- END: ROBUST SEEDING BLOCK ---

class ANN_Diagnoser:
    def __init__(self):
        # Loading the csv data using numpy module
        self.data = np.loadtxt("data_w3_ex1.csv", delimiter=',')
        self.x, self.y = self.cleaned_data()
        self.x_train, self.y_train, \
        self.x_cv, self.y_cv, \
        self.x_test, self.y_test = self.split_data()
        self.train_data, self.cv_data, self.test_data = self.data_mapping_scaling()
        self.nn_models = self.build_models()
        self.nn_train_mses = []
        self.nn_cv_mses = []

    def cleaned_data(self):
        # Separating features(x) and targets(y)
        x = self.data[:, 0]
        y = self.data[:, 1]
        # Changing 1-D arrays into 2D which helps in the code moving forward
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        return x, y

    def split_data(self):
        # training data = 60%, cross-verification data = 20% and testing data = 20%
        x_train, x_, y_train, y_ = train_test_split(self.x, self.y, test_size=.40, random_state=1)
        x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=.50, random_state=1)
        # Deleting temporary variables
        del x_, y_
        return x_train, y_train, x_cv, y_cv, x_test, y_test

    def data_mapping_scaling(self):
        # Mapping
        poly = PolynomialFeatures(1, include_bias=False)
        x_train_mapped = poly.fit_transform(self.x_train)
        x_cv_mapped = poly.transform(self.x_cv)
        x_test_mapped = poly.transform(self.x_test)

        # Feature Scaling
        scaler = StandardScaler()
        x_train_mapped_scaled = scaler.fit_transform(x_train_mapped)
        x_cv_mapped_scaled = scaler.transform(x_cv_mapped)
        x_test_mapped_scaled = scaler.transform(x_test_mapped)
        return x_train_mapped_scaled, x_cv_mapped_scaled, x_test_mapped_scaled

    @staticmethod
    def build_models():
        model_1 = Sequential(
            [
                Dense(25, activation='relu'),
                Dense(15, activation='relu'),
                Dense(1, activation='linear')
            ],
            name='model_1'
        )

        model_2 = Sequential(
            [
                Dense(20, activation='relu'),
                Dense(12, activation='relu'),
                Dense(12, activation='relu'),
                Dense(20, activation='relu'),
                Dense(1, activation='linear')
            ],
            name='model_2'
        )

        model_3 = Sequential(
            [
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(8, activation='relu'),
                Dense(4, activation='relu'),
                Dense(12, activation='relu'),
                Dense(1, activation='linear')
            ],
            name='model_3'
        )
        return model_1, model_2, model_3

    def run_models(self):
        # Running 3 models
        for model in self.nn_models:
            model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=0.001)
            )

            model.fit(
                self.train_data, self.y_train,
                epochs=5000,
                verbose=0
            )

            # Record the training MSEs
            yhat = model.predict(self.train_data)
            train_mse = mean_squared_error(self.y_train, yhat) / 2
            self.nn_train_mses.append(train_mse)

            # Record the CV MSEs
            yhat_cv = model.predict(self.cv_data)
            cv_mse = mean_squared_error(self.y_cv, yhat_cv) / 2
            self.nn_cv_mses.append(cv_mse)

        print("RESULTS:")
        for model_num in range(len(self.nn_train_mses)):
            print(
                "Model {}: Train MSE: {} and CV MSE: {}".format(
                    model_num + 1, self.nn_train_mses[model_num], self.nn_cv_mses[model_num]
                )
            )

        # RESULTS:
        # Model 1: Training MSE: 73.44, CV MSE: 113.87
        # Model 2: Training MSE: 73.40, CV MSE: 112.28
        # Model 3: Training MSE: 44.56, CV MSE: 88.51

        # Model 3 is selected

    def final_test(self, model_num):
        yhat_test = self.nn_models[model_num - 1].predict(self.test_data)
        test_mse = mean_squared_error(self.y_test, yhat_test) / 2

        print("Selected Model: {}".format(model_num))
        print("Training MSE: {}".format(self.nn_train_mses[model_num - 1]))
        print("Cross Validation MSE: {}".format(self.nn_cv_mses[model_num - 1]))
        print("Test MSE: {}".format(test_mse))

        # RESULTS:
        # Selected Model: 3
        # Training MSE: 44.56
        # Cross Validation MSE: 88.51
        # Test MSE: 87.77


# Instance of class ANN_Diagnoser
diagnostor = ANN_Diagnoser()
diagnostor.run_models()
diagnostor.final_test(model_num=3)
