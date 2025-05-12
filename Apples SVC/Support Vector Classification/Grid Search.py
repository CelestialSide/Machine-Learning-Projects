import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality
df = pd.read_csv("apple_quality.csv")
df.dropna(inplace=True)
df["Acidity"] = df["Acidity"].astype(float)

# Create and Normalize features
X = df[["Size", "Weight", "Sweetness", "Crunchiness",
        "Juiciness", "Ripeness", "Acidity"]].copy().to_numpy()

X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Create Label Set
y = df.loc[:, "Quality"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Creates a Support Vector Classifier
clf = SVC(C = 1.0, kernel='rbf')
parameters = {"gamma": np.linspace(0.55, 0.65, num = 9)} # Seems to be 0.6

# Grid Search of gamma variables
grid_search = GridSearchCV(clf, param_grid = parameters, cv = 5)
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_gamma', 'mean_test_score', 'rank_test_score']])

# Grid Search of C Values
parameters = {'C': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]} # Seems to be 20
grid_search = GridSearchCV(clf, param_grid = parameters, cv = 5)
grid_search.fit(X_train, y_train)
C_param = grid_search.best_params_['C']
print(f"Best C value: {C_param}")