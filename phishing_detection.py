import pickle
import numpy as np
import feature_extraction
from sklearn.ensemble import RandomForestClassifier as rfc
#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from flask import jsonify
from sklearn.metrics import accuracy_score

def getResult(url):

     # load the model from disk
    filename = 'finalized_model_n.sav'
    clf = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)

    X_new = []

    X_input = url
    X_new=feature_extraction.generate_data_set(X_input)
    X_new = np.array(X_new).reshape(1,-1)

    try:
        prediction = clf.predict(X_new)
        if prediction == -1:
            return "Phishing Url"
        else:
            return "Legitimate Url"
    except:
        return "Phishing Url"
