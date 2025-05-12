import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Dataset:
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

# Check Grid Search.py for how parameter values were found
clf = SVC(C = 20.0, kernel = 'rbf', gamma = 0.6)

clf.fit(X_train, y_train)
print(f"Score: {clf.score(X_test, y_test):.3f}")

#Create a confusion Matrix of results
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()

plt.savefig("image/apple_confusion.png")
plt.show()
plt.close()