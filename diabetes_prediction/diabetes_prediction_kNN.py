import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.metrics import make_scorer
#from scipy.stats import randint

data = pd.read_csv(sys.argv[1])

# print(data.describe())

#categorical_variables = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
#for var in categorical_variables:
#    plt.figure()
#    sns.countplot(x=var, data=data)
#    plt.title(f'Bar Plot: {var}')
#    plt.show()

data.drop('hypertension', axis=1, inplace=True)
data.drop('heart_disease', axis=1, inplace=True)
data = data[data['gender'] != 'Other']


def encode_categorical_with_label_encoding(data):
    le = LabelEncoder()
    categorical_columns = ['gender', 'smoking_history']
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    return data


def encode_categorical_with_one_hot_encoding(data):
    categorical_columns = ['gender',  'smoking_history']

    for col in categorical_columns:
        encoded_data = pd.get_dummies(data[col])
        data = pd.concat([data, encoded_data], axis=1)
        data = data.drop(col, axis=1)
    return data


X = data.drop("diabetes", axis=1)
y = data["diabetes"]

X = encode_categorical_with_label_encoding(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
columns_to_normalize = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])

knn_classifier = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=6, weights='uniform')
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

micro_f1 = f1_score(y_test, y_pred, average='micro')
print("Micro F1 Score:", micro_f1)

#param_grid = {
#    'n_neighbors': randint(1, 50),
#    'weights': ['uniform', 'distance'],
#    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
#}

#knn = KNeighborsClassifier()

#scorer = make_scorer(f1_score, average='micro')

#random_search = RandomizedSearchCV(knn, param_distributions=param_grid, scoring=scorer, n_iter=10, cv=5)
#random_search.fit(X_train, y_train)

#print("Best Hyperparameters:", random_search.best_params_)

#y_pred = random_search.predict(X_test)

#micro_f1 = f1_score(y_test, y_pred, average='micro')
#print("Micro F1 Score:", micro_f1)
