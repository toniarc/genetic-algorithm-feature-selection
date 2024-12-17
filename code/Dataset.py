import pandas as pd
import numpy as np
import copy
from sklearn import preprocessing
import random
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score

class Dataset:
    """
    A class used to represent a Dataset for machine learning tasks.

    Attributes
    ----------
    df : pandas.DataFrame
        The raw data read from the file.
    df_sampled : pandas.DataFrame
        The sampled data (shuffled if specified).
    label : int
        The index of the label column.
    clf : sklearn.base.BaseEstimator
        The classifier used for training and evaluation.
    X : numpy.ndarray
        The feature matrix.
    y : numpy.ndarray
        The target vector.
    X_train : numpy.ndarray
        The training feature matrix.
    y_train : numpy.ndarray
        The training target vector.
    X_val : numpy.ndarray
        The validation feature matrix.
    y_val : numpy.ndarray
        The validation target vector.
    X_test : numpy.ndarray
        The test feature matrix.
    y_test : numpy.ndarray
        The test target vector.
    features : list
        The list of selected features.
    instances : list
        The list of selected instances.
    ValidationAccuracy : float
        The accuracy on the validation set.
    TestAccuracy : float
        The accuracy on the test set.
    CV : float
        The cross-validation score.

    Methods
    -------
    __init__(file_path, sep, label, divide_dataset=True, header=None)
        Initializes the Dataset object by reading the data file and preprocessing it.
    divide_dataset(classifier, normalize=True, shuffle=True, all_features=True, all_instances=True, evaluate=True, partial_sample=False)
        Divides the dataset into training, validation, and test sets, and optionally evaluates the classifier.
    fit_classifier()
        Trains the classifier on the training set.
    set_validation_accuracy()
        Evaluates the classifier on the validation set and sets the validation accuracy.
    set_test_accuracy()
        Evaluates the classifier on the test set and sets the test accuracy.
    set_CV(folds=5)
        Performs cross-validation and sets the cross-validation score.
    set_train_set(selected_instances)
        Sets the training set to the selected instances.
    set_features(selected_features)
        Sets the features to the selected features.
    set_instances(selected_instances)
        Sets the instances to the selected instances.
    get_validation_accuracy()
        Returns the validation accuracy.
    get_test_accuracy()
        Returns the test accuracy.
    get_CV()
        Returns the cross-validation score.
    """

    # The init method or constructor
    def __init__(self, file_path, sep, label, divide_dataset=True, header=None):
        # Read csv data file
        if (sep == ','):
            df = pd.read_csv(file_path, header=header, sep=',')
        if (sep == ';'):
            df = pd.read_csv(file_path, header=header, sep=';')
        if (sep == ' '):
            df = pd.read_csv(file_path, delim_whitespace=True, header=header)

        # Set raw data attribute
        self.df = df
        self.df_sampled = df

        self.label = label

        # Encode categorical features
        le = preprocessing.LabelEncoder()
        for col in self.df.columns:
            if (self.df[col].dtypes == 'object'):
                list_of_values = list(self.df[col].unique())
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
                le.fit(list_of_values)
                self.df[col] = le.transform(self.df[col])

        # Replcae missing values
        self.df = self.df.replace('?', np.nan)
        self.df = self.df.fillna(self.df.median())
        # df = df.astype('int32')

        if (divide_dataset):
            self.divideDataset()

    def divide_dataset(self, classifier, normalize=True, shuffle=True, all_features=True, all_instances=True,
                      evaluate=True, partial_sample=False):

        # Set classifier
        self.clf = copy.copy(classifier)

        # Shuffle dataset
        if (shuffle):
            self.df = self.df.sample(frac=1, random_state=10)
            self.df_sampled = self.df



        # Divide datset into training/validation/testing
        if (self.label == -1):
            self.X = self.df_sampled.iloc[:, :-1].values
            self.y = self.df_sampled.iloc[:, -1].values
        else:
            selector = [x for x in range(self.df.shape[1]) if x != self.label]
            self.X = self.df_sampled.iloc[:, selector].values
            self.y = self.df_sampled.iloc[:, self.label].values

        X_train = self.X[0:int(0.6 * len(self.X)), :]
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 0.0001
        X_train_normalized = (X_train - mean) / std
        y_train = self.y[0:int(0.6 * len(self.X))]

        X_val = self.X[int(0.6 * len(self.X)):int(0.8 * len(self.X)), :]
        y_val = self.y[int(0.6 * len(self.X)):int(0.8 * len(self.X))]
        X_val_normalized = (X_val - mean) / std

        X_test = self.X[int(0.8 * len(self.X)):, :]
        y_test = self.y[int(0.8 * len(self.X)):]
        X_test_normalized = (X_test - mean) / std

        # Set attribute values
        if (normalize):
            self.X_train = X_train_normalized
            self.y_train = y_train
            self.X_val = X_val_normalized
            self.y_val = y_val
            self.X_test = X_test_normalized
            self.y_test = y_test
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test

        # Confirm instances/features to be used for learning
        if (all_features):
            self.features = np.ones(X_train.shape[1])
        else:
            self.features = np.zeros(X_train.shape[1])
            while (np.sum(self.features) == 0):
                zero_p = random.uniform(0, 1)
                # zero_p = 0.5
                self.features = np.random.choice([0, 1], size=(X_train.shape[1],), p=[zero_p, (1 - zero_p)])
        self.features = list(np.where(self.features == 1)[0])

        if (all_instances):
            self.instances = np.ones(X_train.shape[0])
        else:
            self.instances = np.random.choice([0, 1], size=(X_train.shape[0],), p=[0.5, 0.5])
        self.instances = list(np.where(self.instances == 1)[0])

        # Sample from training split
        if (partial_sample):
            self.instances = np.random.choice(self.X_train.shape[0], partial_sample, replace=False)

        # Train model and evaluate on validation/testing sets
        if (evaluate):
            self.fit_classifier()
            self.set_validation_accuracy()
            self.set_test_accuracy()

    def fit_classifier(self):
        self.clf = self.clf.fit(self.X_train[self.instances, :][:, self.features], self.y_train[self.instances])

    def set_validation_accuracy(self):
        y_pred = self.clf.predict(self.X_val[:, self.features])
        self.ValidationAccuracy = accuracy_score(self.y_val, y_pred)

    def set_test_accuracy(self):
        y_pred = self.clf.predict(self.X_test[:, self.features])
        self.TestAccuracy = accuracy_score(self.y_test, y_pred)

    def set_CV(self, folds=5):
        scores = cross_val_score(self.clf, self.X_train[:][:, self.features], self.y_train[:], cv=folds,
                                 scoring='accuracy')
        self.CV = np.median(scores)

    def set_train_set(self, selected_instances):
        self.X_train = self.X_train[selected_instances]
        self.y_train = self.y_train[selected_instances]

    def set_features(self, selected_features):
        self.features = selected_features

    def set_instances(self, selected_instances):
        self.instances = selected_instances

    def get_validation_accuracy(self):
        return self.ValidationAccuracy

    def get_test_accuracy(self):
        return self.TestAccuracy

    def get_CV(self):
        return self.CV