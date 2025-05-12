import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

df = pd.read_csv("weather_classification_data.csv")
df.dropna(inplace=True)

# Create and Normalize features
X = df[["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
        "Atmospheric Pressure", "UV Index", "Visibility (km)"]].copy().to_numpy()

X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Create Label Set
y = df.loc[:, "Weather Type"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier(verbose = 3, n_jobs = -1)
parameters = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]} # About 3

grid_search = GridSearchCV(clf, param_grid = parameters, cv = 5)
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)


parameters = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90]} # About 10

grid_search = GridSearchCV(clf, param_grid = parameters, cv = 5)
grid_search.fit(X_train, y_train)
n_param = grid_search.best_params_['n_estimators']
print(f"Best amount of trees: {n_param}")
print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])
