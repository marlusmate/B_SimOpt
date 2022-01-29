import numpy as np
import pandas as pd
import SimOptFunctions as fc
import matplotlib.pyplot as plt
import seaborn as sns
print("Bibliotheken erfolgreich importiert")

#load data
df = pd.read_csv('../Data/Concatted/pressure_data.csv')
print("Datensatz erfolgreich geladen: \n", df.head(5))
print("\nFeatures: \n", df.columns)
print("Größe: ", df.shape,"\n")


#Kontext, Labels
drop_col = ['exp', 'Slugflow', 'AirFlow', 'WaterFlow']
lb_in = df['Slugflow']
x_in = df['exp']

#Sensordata
col = list(df.columns)
col = [i for i in col if i not in drop_col]
print("Relevante SensorData: ", col)
print("Label, Kontex: ", drop_col)
df = df[df[col] <= 50][col].dropna()
df = df[df[col] >= -50][col].dropna()
df = df.assign(Slugflow=lb_in)


df_melt = pd.melt(df, id_vars=['Slugflow'], value_vars=col)

#Data Viz
nrows = 5000
method  = 'median'


#median pro Sensor
x_0 = np.asarray([np.median(df[df['Slugflow']==0][c]) for c in col]).reshape((9))
x_1 = np.asarray([np.median(df[df['Slugflow']==1][c]) for c in col]).reshape((9))
x = np.asarray([np.median(df[c]) for c in col]).reshape((9))

plt.figure(1)
plt.clf()
plt.plot(col, x_0, color='#379545', marker='o', linestyle='-')
plt.plot(col, x_1, color='#E89125', marker='o', linestyle='-')
plt.plot(col, x, color='#82CBF2', marker='o', linestyle='dashed')
plt.legend(['Normal', 'Slugging', 'Both'])

#Boxplot
plt.figure(2)
plt.clf()
sns.boxplot(data=df_melt, x='value', y='variable', hue='Slugflow', palette=['#379545', '#E89125'])

#Histograms
plt.figure(4)
for i in np.arange(0,len(col)):
    plt.subplot(3,3,i+1)
    sns.histplot(
        df_melt[df_melt['variable']==col[i]],
        x="value", hue="Slugflow",
        multiple="stack",
        palette=["#307CC0", '#E89125'],
        log_scale=False,
    )
    plt.title(col[i])
plt.show()





