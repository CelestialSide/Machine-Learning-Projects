import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Random Forest Classifier
def RFC(X_train, y_train, X_test, y_test):
        # Initialize Random Forest
    clf = RandomForestClassifier(max_depth=10, n_estimators=40, oob_score=True, verbose=3, n_jobs=-1)
    clf.fit(X_train, y_train)

        # Print Scores
    print(f"Forest Score (Train): {clf.score(X_train, y_train):.3f}")
    print(f"Forest Score (Test): {clf.score(X_test, y_test):.3f}")
    print(f"OOB Score: {clf.oob_score_:.3f}")

    print(mean_squared_error(y_test, clf.predict(X_test)))

        # Confusion Matrix
    cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
    disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
    disp_cm.plot()
    plt.show()

# Bagging Classifier
def bag(X_train, y_train, X_test, y_test):
        # Initialize Bagging Classifier
    clf = SVC()
    clf = BaggingClassifier(clf, n_estimators=63, verbose=3, oob_score=True, n_jobs=-1)
    clf.fit(X_train, y_train)

        # Print Scores
    print(f"Bagging Score (Train): {clf.score(X_train, y_train):.3f}")
    print(f"Bagging Score (Test): {clf.score(X_test, y_test):.3f}")
    print(f"OOB Score: {clf.oob_score_:.3f}")

    print(mean_squared_error(y_test, clf.predict(X_test)))

        # Confusion Matrix
    cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
    disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
    disp_cm.plot()
    plt.show()


df = pd.read_csv("Weather_Cleaned.csv").sample(2000)

    # Create and Normalize features
X = df[["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
        "Atmospheric Pressure", "UV Index", "Visibility (km)"]].copy().to_numpy()

X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

    # Create Label Set
y = df.iloc[:, -1].copy().to_numpy()

    # Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Normal Runs of RFC and Bagging
RFC(X_train, y_train, X_test, y_test)
bag(X_train, y_train, X_test, y_test)

    # Doesn't use test train split, RFC overfits
# RFC(X, y, X, y)
# bag(X, y, X, y)