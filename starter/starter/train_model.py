# Script to train machine learning model.

from nis import cat
from unicodedata import category
from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, label_binarize, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Add code to load in the data.
data = pd.read_csv("../data/modified_v1_census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

def process_data(data, categorical_features, label, training=True):
    x_data = data.drop(columns=[label])
    y_data = data[label]

    cat_x_data = x_data[categorical_features]
    num_x_data = x_data.drop(columns=categorical_features)

    if training==True:
        # one hot encoding categorical features
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_x_data = ohe.fit_transform(cat_x_data.values)
        # X_test = ohe.transform(X_test.values)
        lb = LabelBinarizer()
        y_data = label_binarize(y_data.values, classes=['<=50K', '>50K'])
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_x_data = ohe.transform(cat_x_data.values)
        y_data = label_binarize(y_data.values, classes=['<=50K', '>50K'])

    X_train = np.concatenate([num_x_data, cat_x_data])
    y_train = y_data
    return X_train, y_train, ohe, lb

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
)
# Train and save a model.

