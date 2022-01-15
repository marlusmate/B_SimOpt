#Eval
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics as met
import json

clf = pickle.load(open("../models/model.pkl", "rb"))
X_test = np.load('../Data/FeatureDataTest.npy')
y_test = np.load('../Data/LabelTest.npy')
print("Testdatensätze geladen")
print("Anzahl Instanzen: ", len(y_test))
print('Verteilung Klassen::     Normal: ',np.count_nonzero(y_test==0), '     Slugging: ', np.count_nonzero(y_test==1), '\n')

y_pred = clf.predict(X_test)
pr = met.precision_score(y_test, y_pred)
rc = met.recall_score(y_test, y_pred)
f1 = met.f1_score(y_test, y_pred)
hl = met.hamming_loss(y_test, y_pred)
metr_ls = [['Precision', 'Recall', 'F1-Score', 'Hamming-Loss'], [pr, rc, f1, hl]]
fig = 1
plt.figure(fig)
fig =fig + 1
plt.bar(metr_ls[0], metr_ls[1])
plt.savefig("../Abbildungen/MetricsMLBarplot.png")
#plt.close()

#viz-eval
plt.figure(fig)
fig = fig+ 1
cf_matrix = met.confusion_matrix(y_test, y_pred)
#confusion-matrix
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

ax.xaxis.set_ticklabels(['Normal','Slugging'])
ax.yaxis.set_ticklabels(['Normal','Slugging'])

plt.show()
plt.savefig("../Abbildungen/MetricsMLConfusionmatrix.png")

#write scores to a file
with open("metrics.json", 'w') as outfile:
        json.dump({ "Precision": pr, "Recall": rc, "F1-Score":f1, "Hamming-Loss":hl}, outfile)

with open("../metrics.json", 'w') as outfile:
        json.dump({ "Precision": pr, "Recall": rc, "F1-Score":f1, "Hamming-Loss":hl}, outfile)

print('##Testing abgeschlossen##')
