#import numpy as np
import numpy as np
import NeuralNetwork
from sklearn.ensemble import RandomForestClassifier as rfc
#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from flask import jsonify
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,mean_squared_error

#Importing dataset
data = np.loadtxt("dataset.csv", delimiter = ",")
#Seperating features and labels
X = data[: , :-1]
y = data[: , -1]
np.random.seed(7)
# prepare cross validation
kfold = KFold(10, True, 1)
# enumerate splits
for train, test in kfold.split(data):
        X_train, X_val = X[train], X[test]
        y_train, y_val = y[train], y[test]
        print('train: %s, test: %s' % (len(data[train]), len(data[test])))

#Seperating features and labels
#X = data[: , :-1]
#y = data[: , -1]
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30)

'''
parameters ={'batch_size':[5,10],
            'nb_epoch':[10,20,50],
            'optimizer':['adam','rmsprop','SGD'],
            'learning_rate': list(np.logspace(np.log10(0.001), np.log10(0.1), base = 10, num = 1000))}

model = SimpleNeuralNetwork.SimpleNeuralNetwork()
best_model = GridSearchCV(estimator=model,
                          param_grid=parameters,
                          cv=10,
                         n_jobs=-1,
                         return_train_score=True)

'''
model = NeuralNetwork.NeuralNetwork(hidden_layer_size=2, epochs=400, learning_rate=0.01)
#model = rfc()
model.fit(X_train,y_train)
pr = model.predict(X_val)
print("{0:.2f}".format(model.score(X_val, y_val)))
print("{0:.2f}".format(mean_squared_error(y_val,pr)))
print ("Loss: \n" + str(np.mean(np.square(y_val - pr))))
#print(pr)
#print('Grid Search Best score',best_model.best_score_)
#print('Grid Search Best Parameters', best_model.best_params_)
#print('Execution time',best_model.refit_time_)

# save the model to disk
filename = 'finalized_model_n.sav'
pickle.dump(model, open(filename, 'wb'))
