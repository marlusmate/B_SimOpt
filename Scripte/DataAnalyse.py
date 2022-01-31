import numpy as np
import pandas as pd
import SimOptFunctions as fc
import matplotlib.pyplot as plt
import seaborn as sns
print("Bibliotheken erfolgreich importiert")

#####RAW-DATA###################

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
plt.plot(col, x_0, color="#307CC0", marker='o', linestyle='-')
plt.plot(col, x_1, color='#E89125', marker='o', linestyle='-')
plt.plot(col, x, color='#379545', marker='o', linestyle='dashed')
plt.ylabel("Druck [bar]")
plt.xlabel("Sensor")
plt.legend(['Normal', 'Slugging', 'Both'])

#Boxplot
plt.figure(2)
plt.clf()
sns.boxplot(data=df_melt, x='value', y='variable', hue='Slugflow', palette=["#307CC0", '#E89125'])
plt.xlabel("Druck [bar]")
plt.ylabel("Sensor")

#Histograms
plt.figure(3)
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
    if i >= 6:
        plt.xlabel("Druck [bar]")
    else:
        plt.xlabel("")
plt.show()

#Fisher Score
fs = np.zeros(len(col))
for i in np.arange(0,len(col)):
    fs[i] = fc.fisher_score(df[col[i]], df['Slugflow'])

plt.figure(4)
plt.clf()
plt.barh(col, fs, color='#379545')
plt.title("Fisher-Score")
plt.show()

#expected information gain
igs = np.zeros(len(col))
for i in np.arange(0,len(col)):
    igs[i] = fc.ig(df[col[i]], df['Slugflow'])

plt.figure(5)
plt.clf()
plt.barh(col, igs, color='#379545')
#plt.show()

#medianisierter Zeitlicher Verlauf für 1 Sek. Unterscheidung zw Klassen und Sensoren
time = 1
nrows = time*5000
dict_wave = {}
for c in col:
    dict_wave["{0}".format(c)] ={}
    col_temp = df[c]
    col_1 = col_temp[df['Slugflow'] == 1]
    col_0 = col_temp[df['Slugflow'] == 0]

    dict_wave[c]["{0}".format("Slugging")] = {}
    dict_wave[c]["{0}".format("Normal")] = {}

    dict_wave[c]['Slugging'] = [np.median(col_1[i::nrows]) for i in range(nrows)]
    dict_wave[c]['Normal'] = [np.median(col_0[i::nrows]) for i in range(nrows)]


fignr = 6
for i in np.arange(0,len(col)):
    fig = plt.figure(fignr)
    plt.clf()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle(col[i])
    axs[0].plot(np.arange(0, 5000), dict_wave[col[i]]['Slugging'],  color= '#E89125')
    axs[0].legend(["Slugging"])
    axs[1].plot(np.arange(0, 5000), dict_wave[col[i]]['Normal'], color="#307CC0")
    plt.xlabel("Messung Nr.")
    plt.ylabel("Druck [bar]")
    plt.legend(["Normal"])
    fignr += 1
plt.show()


####### FFT-DATA##########################
x = np.load('../Data/FeatureDataWhole.npy')
y = np.load('../Data/LabelWhole.npy')
y = y.reshape(len(y))

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
plt.figure(16)
nrows = len(np.unique(y))
ncols = len(x[0,:,0])
idx = 1
freq = range(0, len(xs_spec_med))
for i in range(0,2*len(col)):
    plt.subplot(nrows, ncols, idx)
    if i < len(col):
        plt.bar(freq, xn_spec_med[:,i])
    else:
        plt.bar(freq, xs_spec_med[:,i-9])
    idx = idx +1

#fisher score Features
fs_fft = np.zeros((x.shape[2],x.shape[1]))
for sensor in np.arange(0,x.shape[1]):
    spec = x[:,sensor,:]
    fs_temp = np.zeros(2500)

    for ampl in np.arange(1,2500):
        fs_temp[ampl] = fc.fisher_score(spec[:,ampl], y)

    fs_fft.transpose()[sensor] = fs_temp


fs_fft_best =[np.max(fs_fft.transpose()[i]) for i in range(x.shape[1])]
plt.figure(17)
plt.barh(col, fs_fft_best, color='#98BF33')
plt.title("Fisher-Score (max) post FFT")
plt.ylabel("Sensoren")

plt.figure(18)
x_axis = np.arange(len(col))
plt.bar(x_axis - 0.2, fs, 0.4, label = 'preFFT', color='#379545')
plt.bar(x_axis + 0.2, fs_fft_best, 0.4, label = 'postFFT', color='#98BF33')
plt.xticks(x_axis, col)
plt.xlabel("Sensoren")
plt.ylabel("")
plt.title("(Bester) Fisher Score")
plt.legend()
plt.show()


fig = plt.figure(19)
ax = fig.add_subplot(111, projection='3d')

yticks = np.arange(0,x.shape[1])
colors = ['#307CC0', '#82CBF2', '#379545', '#98BF33','#E89125', '#F1DE1E', '#307CC0', '#82CBF2', '#379545']
ax.set_xlabel('Frequenz  [Hz]')
ax.set_ylabel('Sensor Nr.')
ax.set_zlabel('Fisher-Score')
ax.set_yticks(yticks)


for c, k in zip(colors, yticks):
  #Generate the random data for the y=k 'layer'.
  xs = np.arange(0,x.shape[2])
  ys = fs_fft.transpose()[k]

  #providing color to the series
  cs = [c] * len(xs)
  cs[0] = 'c'

  # Plot the 3D bar graph given by xs and ys
  #on the plane y=k with 80% opacity.
  ax.bar(xs, ys, zs=k, zdir='y',
            color=cs, alpha=1)

plt.show()

#expected information gain
igs_fft = np.zeros((x.shape[2],x.shape[1]))
for sensor in np.arange(0,x.shape[1]):
    spec = x[:,sensor,:]
    ig_temp = np.zeros(x.shape[2])
    for mag in np.arange(0,x.shape[2]):
        igs_fft[i] = fc.ig(spec.transpose()[mag], y)

igs_fft_best =[np.max(igs_fft.transpose()[i]) for i in range(x.shape[1])]

plt.figure(20)
plt.clf()
plt.bar(col, igs_fft_best, color='#379545')
#plt.show()