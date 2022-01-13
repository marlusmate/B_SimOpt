# Python Skript zur Vorverarbeitung der "Process Data" #
#Der Große Datensatz wird aufgeteilt in 18 kleinere entsprechend der Zeitstempel der 18 Pressure und Ultrasonic Messungen
    #Zeitstempel der Pressure Messung um 7 Minuten verspätet (vgl. Timestamp Pressure-Messung mit Operation log 0912) , wird korrigiert
    #Sollte also 18 Arrays mit der Größe 60xn  (n= Anzahl nicht erntfernter Spalten) (60 Messreihen für 1Hz Messung über 60s)
#Die Spalten 'FT305/OUT.CV', 'FT302/OUT.CV' (Air Inputs) und 'FT102/OUT.CV', 'FT104/OUT.CV' (Water inputs) werden gemerged
    #Aufgrund p. 18 "Technical Report" jeweils nur ein Ventil in Betrieb (Unterscheidungsmerkmal aus Technical Report)
#Abspeicher in Process/

#####################
#### Import Bibs ####
#####################
import pandas as pd
import numpy as np
import os as os
import datetime
import time

####################
#### Lade Daten ####
####################
path = 'Process/Raw/'                                                   #Pfad zu den Process Dateien
files = os.listdir(path)                                                #listet alle Dateien im Zielordner auf
filename = files[0]                                                     #Da beide Files gleich/#hnlich zu sein scheinen, wird nur mit einem weitergearbeitet
df = pd.read_csv(path + filename, header=1)                             #Laden des Files in pandas Dataframe als "df" (2. Zeile als Spaltenname)

print("####Größe des unbearbeiteten Datensatzes####\n", df.shape)
print(df.head(5))
df = df.drop(0)                                                           #Droppen der Zeile mit Kontext zu Spalten ('Timestamp', 'Air In1',...)
df = df.astype({'FT305/OUT.CV': float, 'FT302/OUT.CV': float, 'FT305/AI2/OUT.CV': float,
                'PT312/OUT.CV': float, 'FT102/OUT.CV': float, 'FT104/OUT.CV': float, 'FT102/AI3/OUT.CV': float,
                'FT102/AI2/OUT.CV': float, 'PT417/OUT.CV': float, 'PT408/OUT.CV': float,
                'FIC302/PID1/OUT.CV': float, 'FIC302/PID1/SP.CV': float, 'FIC302/PID1/PV.CV': float,
                'FIC301/PID1/OUT.CV': float, 'FIC301/PID1/SP.CV': float, 'FIC301/PID1/PV.CV': float,
                'FIC102/PID1/OUT.CV': float, 'FIC102/PID1/SP.CV': float, 'FIC102/PID1/PV.CV': float,
                'FIC101/PID1/OUT.CV': float, 'FIC101/PID1/SP.CV': float, 'FIC101/PID1/PV.CV': float})


#######################
#### PREPROCESSING ####
#######################

#Droppen unwichtiger Spalten (aus Tabelle 2, p. 6 "Technical Report")
drop_cols = df.columns[11:20]                                           #Definition der zu entfernenden Spalten (FT404-LI101 aus Tab.2)
df = df.drop(labels= drop_cols, axis=1)
print("\n####Entfernte Spalten####\n", drop_cols)

df.rename( columns={'Unnamed: 0':'Timestamp'}, inplace=True )           #Spalte für Messzeitpunkte bennen

print("\n####Verbliebene Spalten####\n", df.columns)
print("\n####Größe des neuen Datensatzes####\n", df.shape)
print(df.head(5))

#Extrahieren der Uhrzeit (Zeitstempel) der einzelnen Messreihen aus Process data
#temp_ts = df.iloc[:]['Timestamp'].str[11:16]
temp_ts = df['Timestamp'].copy()
temp_ts = pd.DataFrame(temp_ts)
temp_ts['Timestamp'] = temp_ts.iloc[:]['Timestamp'].str[11:16]
print("type temps_ts: ", type(temp_ts))
print("Bsp temp_ds: ", temp_ts.head(5))

ts_df = pd.to_datetime(temp_ts['Timestamp'], format="%H:%M")
#ts_df = ts_df.date.dt.hour
print("type ts_df: ", type(ts_df[1]))
print("Bsp ts_df: ", ts_df.head(5))


#Rauspicken der Daten welche sich mit anderen Datensätzen überschneiden (anhand Uhrzeit und Name)
#Und mergen der Valve-Input Values (siehe p.18 "Technical Report")
ts = pd.read_csv('ExpTimestamp.csv')                                    #in 'ExpTimestamp.csv' Uhrzeit und Air-Water-Flow..
print("\ntype ts[Timestamp]: ", type(ts['Timestamp']))                        #...Verhältnis der Pressure Messungen abgespeichert
print("Bsp ts[Timestamp]: ", ts['Timestamp'][1])

ts_exp = pd.to_datetime(ts['Timestamp'], format="%H:%M")             #Auftrennen Zeitpunkt und Flow-Einstellungen, #Überführung von string zu DateTime (um Nachher die 7Minuten Zeitverschiebung zu korrigieren)
print("\ntype ts_exp: ", type(ts_exp[1]))
print("Bsp ts_exp: ", ts_exp[1])

time_shift = pd.Timedelta(7, unit='minutes')          #Pressuredate 7 Minuten zu spät abgespeichert (laut ts)
print("\ntype time_shift: ", type(time_shift))
print("Bsp time_shift: ", time_shift)

#Zeitverzug ausgleichen
for ind in range(len(ts_exp)):
    ts_exp[ind] = ts_exp[ind] - time_shift
print("\ntype_exp raw: ", type(ts_exp[1]))
#Stunde auswählen (aus komplettem Datum)
#for ind in range(len(ts_exp)):
    #ts_exp[ind] = ts_exp[ind].time()
print("\ntype ts_exp: ", type(ts_exp[1]))
print("Bsp ts_exp: ", ts_exp[1])

nm_exp = ts['ExpNr']
print("\ntype nm_exp: ", type(nm_exp[1]))

for j in range(len(nm_exp)):
    #Ziehen der Messreihen die in der gleichen Minute wie die (jeweiligen) pressure Daten aufgenommen wurden
    #ind_j = np.zeros(len(df))
    ind_j = ts_exp[j] == ts_df
    print("\nn ind_j: ", sum(ind_j))
    df_temp = df.loc[ts_exp[j] == ts_df]                                          #Extrahiere aus df alle Reihen entsprechend index ind_j, und alle Spalten (:)
    print("\nbetrachtete Uhrzeit: ", ts_exp[j])
    print("Datentyp df_temp: ", type(df_temp))
    print("Länge extrahierter Datensatz: ", len(df_temp))
    print("Bsp Type 'FIC301/PID1/SP.CV': ", type(df_temp.iloc[0]['FIC301/PID1/SP.CV']))

    #Mergen der Valve-Inputs (wahrscheinlich irrelevant, da valves zwischen den stationären zuständen switchen sollten)
    #FT305, FT302 zu 'AirIn' (da aufsummieren nicht empfohlen) (Entscheidungswert: 'FIC301/PID1/SP.CV' > 150)
    df_temp[ 'AirIn'] = df_temp['FT305/OUT.CV'].values[:]
    for m in range(len(df_temp)):
        if df_temp['FIC301/PID1/SP.CV'].values[m] >= 150:
            df_temp['AirIn'].values[m] = df_temp['FT302/OUT.CV'].values[m]
            print(df_temp['FT305/OUT.CV'].values[m], "ausgetauscht mit ", df_temp['FT302/OUT.CV'].values[m])

    df_temp = df_temp.drop(['FT302/OUT.CV', 'FT305/OUT.CV'], axis=1)
    print("Versuchsreiche:", nm_exp[j], "Air-Valves 'FT302/OUT.CV', 'FT305/OUT.CV' zu 'AirIn' gemerged: \n", df_temp['AirIn'].head(5), "\n")

    #Ft102, FT104 zu 'WaterIn' (da aufsummieren nicht empfohlen)
    df_temp[ 'WaterIn'] = df_temp['FT104/OUT.CV'].values[:]
    for n in range(len(df_temp)):
        if df_temp.iloc[n]['FIC102/PID1/SP.CV'] >= 1:
            df_temp['WaterIn'].values[n] = df_temp['FT102/OUT.CV'].values[n]
            print(df_temp['FT104/OUT.CV'].values[n], "ausgetausch mit ", df_temp['FT102/OUT.CV'].values[n])
    df_temp = df_temp.drop(['FT104/OUT.CV', 'FT102/OUT.CV'], axis=1)
    print("\nVersuchsreiche:", nm_exp[j], " Water-Valves 'FT104/OUT.CV', 'FT102/OUT.CV' zu 'WaterIn' gemerged: \n", df_temp['WaterIn'].head(5))

    #Droppen der Spalten der Steuerungstechnik Variablen (FIC302/PID1/OUT.CV, etc...)
    df_temp = df_temp.drop(['FIC302/PID1/OUT.CV', 'FIC302/PID1/SP.CV', 'FIC302/PID1/PV.CV',
                'FIC301/PID1/OUT.CV', 'FIC301/PID1/SP.CV', 'FIC301/PID1/PV.CV',
                'FIC102/PID1/OUT.CV', 'FIC102/PID1/SP.CV', 'FIC102/PID1/PV.CV',
                'FIC101/PID1/OUT.CV', 'FIC101/PID1/SP.CV', 'FIC101/PID1/PV.CV'], axis=1)
    df_temp = df_temp.drop(['Timestamp'], axis=1)                                         #passiert alles in einer Minute, höher ist die Auflösung der Process Data Zeit nicht (also steht da theoretisch immer die gleiche Zeit)
    print("Head fertig vorverarbeiteter Datensatz Process", nm_exp[j], ":\n", df_temp.head(5))


    #Abspeichern der einzelnen Files
    #df_temp = df_temp.reset_index(drop=True, inplace=True)
    print("\nDatentyp df_temp: ", type(df_temp))
    df_temp.to_csv('Process/Processdata'+ nm_exp[j] + '.csv')
    print("\n###Datensatz: Processdata", nm_exp[j], " vorverarbeitet und abgespeichert####\n" )

print("\n>>>>>> VORVERARBEITUNG PROCESS DATA ABGESCHLOSSEN <<<<<<")



