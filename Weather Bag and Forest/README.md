# Weather Classification : Random Forest Classification Versus Bagging Classification

This project means to compare two Classification algorithms, Bagging Classification (Using Support Vector Classification)
and Random Forest Classification

## Dataset

- https://www.kaggle.com/datasets/nikhil7280/weather-type-classification
- Name : Weather Type Classification
- Author : Nikhil Narayan
- All features were utilized
  - Label Variable was `Weather Type` 

## File Overview

- `bagging_confusion.png` : Confusion matrix of Baggin gClassifier trained with Train Test Split
- `bagging_full_confusion.png` : Confusion matrix of Bagging Classifier trained on full dataset with no split
- `forest_confusion.png` : Confusion matrix of RFC trained with Train Test Split
- `forest_full_confusion.png` : Confusion matrix of RFC trained on full dataset with no split
- `Bagging SVC Optimizing.py` : Finds `n_estimator` using a Grid Search
- `Forest Optimizing.py` : Finds both `n_estimator` and `max_depth` using a Grid Search
- `Output.py` : This program runs both algorithm for comparison
- `Cleaning.py` : This program cleans the dataset
- `Weather_Cleaned.csv` : Cleaned Dataset that was used
- `weather_classification_data.csv` : Raw Dataset from Kaggle
- `README.md` : Your reading it right now!

## Model Overview
Two strategies were used for comparison, using a test train split and training on the entire datasets.

- Parameters :
  - Bagging Classification
    - `n_estimator` : 63
    - `Classification Base` : Support Vector Classifier
  - Random Forest Classification
    - `max_depth` : 10
    - `n_estimator` : 40

## Results
Random Forest Classification tends to do better than the Bagging Classifier in general, specifically RFC averaged around a 90-93% 
accuracy which is quite good. In comparison the Bagging Classifier would get around 89-90%.

Random Forest Classification

![Confusion Matrix](Bagging%20Vs%20Forest%20Weather/image/forest_confusion.png)

Bagging Classification

![Confusion Matrix](Bagging%20Vs%20Forest%20Weather/image/bagging_confusion.png) 

In comparison, when the test train split was dropped the Random Forest Classifier would jump up to right below 100%, but it's out of bag
score would stagnate at around 90%, showing that the RFC overfits when trained on the entire result. In comparison the Bagging Classifier
ended with a similar score

Random Forest Classification

![Confusion Matrix](Bagging%20Vs%20Forest%20Weather/image/forest_full_confusion.png)

Bagging Classification

![Confusion Matrix](Bagging%20Vs%20Forest%20Weather/image/bagging_full_confusion.png)
