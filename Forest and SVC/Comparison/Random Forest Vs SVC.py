import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt

df = pd.read_csv("star_class_data.csv")
df.dropna(inplace=True)

# Create and Normalize features
X = df[["Vmag", "Plx", "e_Plx", "B-V", "Amag"]].copy().to_numpy()

X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Create Label Set
y = df.loc[:, "TargetClass"].copy().to_numpy()

# Split test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Support Vector Classifier
clf = SVC(C = 8.0, kernel = 'rbf', gamma = 0.24)
clf.fit(X_train, y_train)

# Random Forest Classifier
alf = RandomForestClassifier(max_depth = 7, n_estimators = 20, oob_score = True, verbose = 3)
alf.fit(X_train, y_train)

# Outputs SVC Scores and Confusion Matrix
print(f"\nSVC Score (Train): {clf.score(X_train, y_train):.3f}")
print(f"SVC Score (Test): {clf.score(X_test, y_test):.3f}")

cm = confusion_matrix(y_test, clf.predict(X_test), normalize= "true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.savefig("image/SVC_confusion.png")
plt.show()

# Outputs Forest Scores and Confusion Matrix
print(f"\nForest Score (Train): {alf.score(X_train, y_train):.3f}")
print(f"Forest Score (Test): {alf.score(X_test, y_test):.3f}")

cm = confusion_matrix(y_test, alf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=alf.classes_)
disp_cm.plot()
plt.savefig("image/forest_confusion.png")
plt.show()



