#Bibs
import numpy as np

print('Bibliotheken erfolgreich geladen\n')



#load features
X = np.load('../Data/FeatureDataTrain.npy')
y = np.load('../Data/LabelTrain.npy')
print('Features, Label erfolgreich geladen')
print('Shape Feature Set: ', y.shape)
print('Verteilung Klassen::     Normal: ',np.count_nonzero(y==0), '     Slugging: ', np.count_nonzero(y==1), '\n')


#Train
X_train = X
y_train = y
#hier fehlt noch sh für random search

#model
from sklearn.tree import DecisionTreeClassifier as DTC
clf = DTC()

#random search
import random as rd
score = 0
seed = 22
max_iter = 20
k = 5
splitter = ["best", "random"]
max_depth = range(0,10)

for i in range(0,max_iter):
    sp = splitter[rd.randint(0,1)]
    md = max_depth[rd.randint(0,9)]
    sc_temp = np.zeros(k)
    #cross-val
    for j in range(0,k):
        trval_dtemp = np.split(X_train[:-(len(X_train)%k)],k)
        trval_ltemp = y_train[:len(trval_temp)]
        val_dtemp = trval_dtemp[j]
        val_ltemp = trval_ltemp[j]
        train_dtemp = []
        train_ltemp = []

        for k in range(0,k):
            if k != j:
                train_dtemp = np.append(train_dtemp, trval_temp[k])
                train_ltemp = np.append(train_ltemp, trval_temp[k])
        clf.fit(train_dtemp, train_ltemp)
        sc_temp[j] = clf.score

    mscore = np.mean(sc_temp)
    if mscore > score:
        score = mscore

print('Model erfolgreich trainiert\n')


#Save
import pickle
import os

#create dir (?)
if not os.path.isdir("../models/"):
    os.mkdir("models")

filename = '../models/model.pkl'
pickle.dump(clf, open(filename, 'wb'))


print('##Training erfolgreich abgeschlossen##\n')
print('##Model erfolgreich abgespeichert')
