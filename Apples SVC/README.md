# Apple Quality : Support Vector Classification

This project uses a Radial Support Vector Classification function in order to classify Apples based on their quality.

## Dataset

- https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality
- Name : Apple Quality
- Author : Nidula Elgiriyewithana

## File Overview

- `Apples!.py` : Cleans and Normalizes Dataset and then evaluates using SVC.
- `Grid Search.py` : Finds optimal `C` and `Gamma` variables for SVC through use of Grid Search.
- `apple_quality.csv` : Full Dataset from Kaggle.
- `apple_confusion.png` : Picture of the Confusion Matrix
- `README.md` : Your reading it right now!

## Model Overview
Dataset was split into 70% train and 30% test

- Features :
  - `Size`
  - `Weight`
  - `Sweetness`
  - `Crunchiness`
  - `Juiciness`
  - `Ripeness`
  - `Acidity`
- Label :
  - `Quality`

- Parameters :
  - `C` : 20
  - `Kernel` : Radial Basis Function (rbf)
  - `Gamma` : 0.6

## Results
The overal accuracy was 0.906 or 90.6% which is quite good, delving into the confusion matrix (Shown Below) we can see that the majority of the
error stems from false negatives as good apples are labeled as bad. This is better than the alternative as in a real world scenerio it would be
better for a few good apples to be thrown away in comparison to sending bad apples to market.

![Confusion Matrix](Support%20Vector%20Classification/image/apple_confusion.png)


