# Python Skript zum Labeln und Concaten der UT-Daten
#labeln und zusammenfassen aller Ultraschallmessungen zur Analyse

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

### Datenverzeichnisse laden ###
files_ut = os.listdir('RawData/Ultrasonic/')
files_ut.remove('Raw')
print("Liste Files Ultrasonic: ", files_ut)


### Laden und Labeln ###
    #PRESSURE
df_ut = pd.DataFrame()
print("\n##Laden der einzelnen Files##\n")
for file_ut in files_ut:
    df = pd.read_csv('Ultrasonic/' + file_ut, header=0)
    print("Größe Ultrasonic Datensatz: ",df.shape)

    #Extraktion der Air und Wt Parameter (und Label (lb) aus labels-matrix), kann bestimmt schöner gemacht werden
    exp = file_ut[:-4]
    exp = exp.split('_')
    print("\nExperiment Eckdaten: ", exp)
    ar = exp[1]
    wt = exp[2]
    lb = labels.loc[ar, wt]
    print("Values: Ar, WT: ", ar, wt)
    print("type: Ar, WT: ", type(ar), type(wt))
    print("Label aus Tabelle: ", labels.loc[ar, wt])


    #Labeling
    df = df.assign(Slugflow=np.ones((len(df),1), dtype=int)*lb,
                   AirFlow = np.ones((len(df), 1), dtype=int) * int(exp[1]),
                    WaterFlow = np.ones((len(df), 1), dtype=int) * int(exp[2]))

    print("Slug-, Air- ,Waterflow Spalte: \n", df['Slugflow'][0:4])

    #Concaten
    df_ut = df_ut.append(df, ignore_index=True )


print("\nGröße Zusammengefasster Datensatz: ", df_ut.shape)
print("Bsp Head neuer allg. Ultrasonic: \n", df_ut.head(10))

#Speichern
df_ut.to_csv('Concatted/Ultrasonic_data.csv', index=False)
print("\n############### Ultraschall Datensätze erfoglreich zusammengefasst ###############")
