# Python Skript zum Labeln und Concaten der Pressure Daten
#labeln und zusammenfassen aller Druckmessungen zur Analyse

################
#### IMPORT ####
#################

### Bibs ###
import os
import pandas as pd
import numpy as np
import matplotlib as mlp
import seaborn as sns

#### Datalabel ###
#Daten am besten so laden dass sie gleichzeitig gelabelt werden
#Def Label-matrix, 1 == SLUGGING, 0 == NORMAL
cl_wt = ['01', '05', '1', '2', '35']
cl_ar = ['20', '50', '100', '200']
labels = pd.DataFrame([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], cl_ar, cl_wt )
print("Label Tabelle: ", labels)
print("type cl listen: ", type(cl_wt))

#Lade Daten
#Daten Laden und dann Sortieren, Process Data ist in anderer Reihenfolge als der Rest abgespeichert

### Datenverzeichniss laden ###
files_pre = os.listdir('../Data/RawData/Pressure/')
files_pre.remove('Raw')
print("\nListe Files Press: ", files_pre)


### Laden und Labeln ###
    #PRESSURE
df_pres = pd.DataFrame()
print("\n##Laden der einzelnen Files##\n")
for file_pre in files_pre:
    df = pd.read_csv('Pressure/' + file_pre, header=0)
    print("Größe Pressure Datensatz: ",df.shape)

    #Extraktion der Air und Wt Parameter (und Label (lb) aus labels-matrix), kann bestimmt schöner gemacht werden
    expr = file_pre[:-4]
    exp = expr.split('_')
    df['exp'] = expr
    print("\nExperiment Eckdaten: ", exp)
    ar = exp[1]
    wt = exp[2]
    lb = labels.loc[ar, wt]
    print("Values: Ar, WT: ", ar, wt)
    print("type: Ar, WT: ", type(ar), type(wt))
    print("Label aus Tabelle: ", labels.loc[ar, wt])


    #Labeling
    df = df.assign(Slugflow=np.ones((len(df),1), dtype=int)*lb,
                   AirFlow=np.ones((len(df),1), dtype=int)*int(exp[1]),
                   WaterFlow=np.ones((len(df),1), dtype=int)*int(exp[2]))
    print("Slug-, Air-, Waterflow Spalte: \n", df[['Slugflow', 'AirFlow','WaterFlow']][0:4])

    #Concaten
    df_pres = df_pres.append(df, ignore_index=True )


print("\nGröße Zusammengefasster Datensatz: ", df_pres.shape)
print("Bsp Head neuer allg. Pressure Datensatz: \n", df_pres.head(10))

#Speichern
df_pres.to_csv('../Data/Concatted/pressure_data.csv', index=False)
print("\n############### Druck Datensätze erfoglreich zusammengefasst ###############")
