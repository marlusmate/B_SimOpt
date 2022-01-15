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
clf.fit(X_train, y_train)

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
