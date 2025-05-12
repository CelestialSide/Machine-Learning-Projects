import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv("weather_classification_data.csv").sample(2000)

# Create and Normalize features
X = df[["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
        "Atmospheric Pressure", "UV Index", "Visibility (km)"]].copy().to_numpy()

X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Create Label Set
y = df.loc[:, "Weather Type"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = SVC()
clf = BaggingClassifier(clf, verbose = 3, oob_score=True, n_jobs = -1)
parameters = {"n_estimators": [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]}
clf.fit(X_train, y_train)

grid_search = GridSearchCV(clf, param_grid = parameters, cv = 5)
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)

print(score_df[['param_n_estimators', 'mean_test_score', 'rank_test_score']])