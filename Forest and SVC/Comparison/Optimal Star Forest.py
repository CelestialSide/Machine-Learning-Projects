import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

df = pd.read_csv("star_class_data.csv")
df.dropna(inplace=True)

# Create and Normalize features
X = df[["Vmag", "Plx", "e_Plx", "B-V", "Amag"]].copy().to_numpy()

X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Create Label Set
y = df.loc[:, "TargetClass"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier(verbose = 3, n_jobs = -1)
parameters = {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]} # Seems to be 7

#grid_search = GridSearchCV(clf, param_grid = parameters, cv = 5)
#grid_search.fit(X_train, y_train)
#score_df = pd.DataFrame(grid_search.cv_results_)
#print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])

parameters = {'n_estimators': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]} # Seems to be around 20

grid_search = GridSearchCV(clf, param_grid = parameters, cv = 5)
grid_search.fit(X_train, y_train)
n_param = grid_search.best_params_['n_estimators']
print(f"Best amount of trees: {n_param}")
