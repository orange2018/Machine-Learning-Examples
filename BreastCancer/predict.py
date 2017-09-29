import joblib
import numpy as np

clf = joblib.load("results/classifier.pkl")
print(clf)

def predict(features):
	return np.asscalar(clf.predict(features))