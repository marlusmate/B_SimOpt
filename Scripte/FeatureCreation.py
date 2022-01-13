#Bibs
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

import SimOptFunctions as fc

print("Bibliothken erfolgreich Importiert\n")

#Data
df = pd.read_csv('../Concatted/pressure_data.csv')
print("Datensatz erfolgreich geladen: \n", df.head(5))
print("\nFeatures: \n", df.columns)
print("Größe: ", df.shape,"\n")
#short data for test purposes
#df = df.iloc[:len(df)//2][:]

#Kontext, Labels
drop_col = ['exp', 'Slugflow', 'AirFlow', 'WaterFlow']
#Sensordata
col = list(df.columns)
col = [i for i in col if i not in drop_col]
#colx = ['P4-B14', 'P5-B13','P6-B12','P7-B11','P8-B10','P13-B20','P14-B08','P15-B09','P17-B05','Slugflow', 'id', 'freq']

#shorten data - feature Creation
nrows = 5000
method  = 'fft_mag'
lb_in = df['Slugflow']
x_in = df['exp']

x, y = fc.shorten_df(df, lb_in, x_in, nrows, method)

print('\nFeatures, Labels erfolgreich erstellt')
print('Shape Feature: ', x.shape)
print('Verteilung Klassen::     Normal: ',np.count_nonzero(y==0), '     Slugging: ', np.count_nonzero(y==1))

#eventl weiter ausführen

#feature eval, selection?
#divide set by clas
xn = x[(y.reshape(len(y))==0),:,:]
xs = x[(y.reshape(len(y))==1),:,:]
#medians for each sensor spectrum
xn_spec_med= np.empty((len(xn[0,0,:]), len(col)))
xs_spec_med= np.empty((len(xs[0,0,:]), len(col)))

for i in range(0,len(col)):
    xn_spec_med[:,i] = [np.median(xn[:,i,j]) for j in range(0,len(xn[0,0,:]))]
    xs_spec_med[:,i] = [np.median(xs[:,i,j]) for j in range(0,len(xs[0,0,:]))]

#viz
import matplotlib.pyplot as plt
nrows = len(np.unique(y))
ncols = len(x[0,:,0])
idx = 1
freq = range(0, len(xs_spec_med))
for i in range(0,2*len(col)):
    plt.subplot(nrows, ncols, idx)
    if i < len(col):
        plt.plot(freq, xn_spec_med[:,i])
    else:
        plt.plot(freq, xs_spec_med[:,i-9])
    idx = idx +1

#Datensatz ml-bearbeitbar transformieren
X = x.reshape(len(y), len(x[0,0,:])*len(x[0,:,0]))
y= y.reshape(len(y))
print('Feature Set, Label erfolgreich transformiert: \n', X.shape, y.shape)

#DataS splitten
from sklearn.model_selection import train_test_split
seed = 69
tr_sz = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=tr_sz, random_state = seed)
print('Datensatz gesplittet')
print('Verteilung Klassen Train::     Normal: ',np.count_nonzero(y_train==0), '     Slugging: ', np.count_nonzero(y_train==1), '\n')
print('Verteilung Klassen Test::     Normal: ',np.count_nonzero(y_test==0), '     Slugging: ', np.count_nonzero(y_test==1), '\n')


#save
#Finales Test Set hier abschneiden ?
print("Shape Features (Train, Test): \n", X_train.shape, "\n", X_test.shape, "\n")
print("Shape Labels (Train, Test): \n", y_train.shape, "\n", y_test.shape, "\n")

np.save('../Data/FeatureDataTrain.npy', X_train)
np.save('../Data/FeatureDataTest.npy', X_test)
np.save('../Data/LabelTrain.npy', y_train)
np.save('../Data/LabelTest.npy', y_test)
print('##TrainingData, Label erfolgreich gespeichert##')

print('x')