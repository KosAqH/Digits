# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pandas as pd
import joblib

if __name__ == "__main__":
    df = pd.read_csv("Data\\train.csv")
    
    seed = 0
    X = df.iloc[:, 1:].copy().values
    y = df['label'].copy().values

    svc = SVC(C=0.1, gamma=1, kernel="poly")
    svc.fit(X, y)
    joblib.dump(svc, "svc.joblib")
    print("Saved SVC model")

    knn = KNeighborsClassifier(metric="minkowski", n_neighbors=5, weights="distance")
    knn.fit(X, y)
    joblib.dump(knn, "knn.joblib")
    print("Saved KNN model")
    
