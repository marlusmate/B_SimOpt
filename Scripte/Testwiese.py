#### Python Skript zur Visualisierung der Messdaten ###


########
#Import#
########
#Bibs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("Bibliothken erfolgreich Importiert\n")


##############
###PRESSURE###
##############

#Datein
df = pd.read_csv('Concatted/pressure_data.csv')
print("Datensatz erfolgreich geladen: \n", df.head(5))
print("\nFeatures: \n", df.columns)
print("\nGröße: ", df.shape)

##dataprep
drop_col = ['exp', 'Slugflow', 'AirFlow', 'WaterFlow']
col = list(df.columns)
col = [i for i in col if i not in drop_col]
##Allg. Metriken
df = df[df['P4-B14'] <= 50]
df = df[df['P5-B13'] >= -50]

metr = df[col].describe()
#metr.to_excel("PressureDescriptiveMetrics.xlsx", sheet_name="Sheet_LongData_1")

#Bisheriger Datensatz zu Groß zur Analyse -> muss zusammengefasst werden (mit "shorten" function
#FUNCTIONS
def shorten(d_in, col_ex, rows, method):
    i=0
    j=0
    df_temp = d_in[col_ex]
    df_temp = df_temp[:-(len(df_temp)%rows)]
    ind = d_in.iloc[np.arange(0,len(df_temp),rows)]['Slugflow']
    if method != "fft":
        df_sh = pd.DataFrame(np.ones((int(len(df_temp)/rows),len(col_ex))), columns=col_ex)
    if method == "fft":
        df_sh = pd.DataFrame(np.ones((len(d_in), len(col_ex))), columns=col_ex)
        ind = d_in['Slugflow']
        ts = np.zeros((len(d_in),1))
    for i in np.arange(0,len(df_temp),rows):
        if method == "mean":
            df_sh.iloc[j] = [np.mean(df_temp.iloc[i:i+rows-1][c]) for c in col_ex] #Loop Durch Spalten
        if method == "median":
            df_sh.iloc[j] = [np.median(df_temp.iloc[i:i + rows - 1][c]) for c in col_ex]
        if method == "fft":
            ts[i:i + rows] = j
            dt = 1/5000
            t= np.arange(0,(dt*rows), dt)
            fhat= [np.fft.fft(df_temp.iloc[i:i + rows][c]) for c in col_ex]
            PSD = (fhat* np.conj(fhat)/rows).real
            #freq = (1/dt*(len(t))) * np.arange(n)
            #L = np.arange(1, np.floor(n/2), dtype='int')
            df_sh.iloc[i:i + rows]=PSD.transpose()
        j= j+1 #Loop durch kleinen, zu erstellenden Datensatz
    df_sh = df_sh.assign(Slugflow=ind.tolist())
    if method == "fft":
        df_sh = df_sh.assign(SampleNr=ts)
    return df_sh

def fisher_score(fe, c):
    m_f = np.mean(fe)
    m_f_c = [np.mean(fe[i==c]) for i in set(c)]
    diff_sq = np.square([m_f_c-np.ones(len(m_f_c))*m_f])
    v_f_c=  [np.var(fe[i==c]) for i in set(c)]
    return sum(len(fe)*diff_sq)/sum(len(fe)*np.square(v_f_c))

# FFt
def fft(f_in, n):
    Fs = 5000  # Sampling Rate, Abtastrate 5kHz [1/s]
    tstep = 1 / Fs  # Zeit zwischen Abtastung [s]
    f0 = []  # ? Signal Frequenz, unbekannt?
    # n = 1 #Betrachtungsdauer [s]
    t = np.arange(0, n, tstep)  # Messzeitpunkte

    ns = len(t)  # Anzahl Samples [-]
    L = np.arange(1, np.floor(ns / 2), dtype='int')

    fhat = np.fft.fft(f_in)
    fhat = np.hstack((np.zeros(1), fhat[L]))
    # FastFourierTransf., ergibt n=len(f_in) komplexe Fourierkoeffizienten [-],
    # Magnitude(Gewichtigkeit/betrag) & Phase(sin/cos Verhältnis)  (?)
    # die (jeweils) aufsummiert werden müssten um das vorgegebene Signal darzustellen
    # (Deswegen brauchen wir auch mindestens n Frequenzen  (rekonstruktion))

    psd = fhat * np.conj(fhat) / ns
    psd = np.hstack((np.zeros(1), psd[L]))
    # Berechnung Power Spectrum (Leistungsdichte über dem Spektrum der Freq)
    # "fhat*np.conj(fhat)" ergibt Quadrat des Betrages des img Vektors (jeder Zeile/Frequenz)
    # Gesamte Leistung des Signal über die Bandbreite der n Frequenzen verteilt ("/ns")
    # (Das Integral dieses Graphen ergibt demnach wieder die Leistung des Signal)
    # Komplex konjug. des Vektor ergibt Stärke der jeweiligen Freq. [W?]

    mag = np.abs(fhat) / ns
    mag = 2 * mag[0:int(ns / 2)]
    mag[0] = mag[0] / 2

    freq = (1 / (tstep * ns)) * np.arange(ns)
    freq = np.hstack((np.zeros(1), freq[L]))
    # Berechnung des zum psd-vektor korrespondierenden Frequenz-vektor

    # post process
    f_out = np.array([fhat, psd, mag, freq])
    f_out = pd.DataFrame(f_out.transpose(), columns=['fhat', 'psd', 'mag', 'freq'])

    return f_out

#Zusammenfassen Datensatz
n_rows = 5000
df_t = shorten(df, col, n_rows, "mean")
print("\nMextrics OG Data: \n", df.describe())
print("Metrics gekürtzte Data: \n", df_t.describe())
#ummodeln
data_t = pd.melt(df_t, id_vars= 'Slugflow', value_vars= col, var_name="Messpunkt")
print("Verhältnis normal/slugging: ", np.count_nonzero(df_t['Slugflow']==0)/np.count_nonzero(df_t['Slugflow']==1))
print("\nUmgemodelten Datensatz zur visualisierung erstellt, data")
print("Neuer Datensatz: \n", data_t.sample(7), "\n Shape: ", data_t.shape)
#Erstellen ParrallelDatensatz (auch kurz) FF-transformiert
df_f= fft(df_t, 'Slugflow')
data_f = pd.melt(df_f, id_vars= 'Slugflow', value_vars= col, var_name="Messpunkt")
##fft2
dt = 1/5000
n = len(df.iloc[:5000]['P4-B14'])
fhat=  np.fft.fft(df.iloc[:5000]['P4-B14'],n)
PSD = fhat * np.conj(fhat)/n
freq= (1/dt*n)*np.arange(n)
L = np.arange(1, np.floor(n/2), dtype='int')
plt.figure()
plt.plot(freq[L], PSD[L])

#dataeval
n_f = 1
#Verteilung Klassen
n_n = np.count_nonzero(df_t['Slugflow']==0)
n_s = np.count_nonzero(df_t['Slugflow']==1)
plt.figure(num=n_f)
n_f = n_f+1
plt.bar(["Normal", "Slugging"], [n_n, n_s])
plt.ylabel("Instanzen")
print("\nAnzahl Instanzen je Klase; Normal: ", n_n, " Slugging: ", n_s)
print("Verhältnis normal/slugging: ", n_n/n_f)

#Bso Sensor genauer Anschauen
#Bestimmung relevanter Softsensoren über Filter Methods,
#FISHER SCORE
##df_t
fish_sc = pd.DataFrame([fisher_score(df_t[i],df_t['Slugflow']) for i in col], columns=['0', '1'])
fish_sc['Messpunkt'] = col
fish_sc_melt = pd.melt(fish_sc, id_vars='Messpunkt', value_vars=['0', '1'], var_name='Slugging')
fs_n= sns.barplot(x='Messpunkt', y='value', hue='Slugging', data=fish_sc_melt)
fs_n.set_title("Fisher Score pre FFT")
plt.savefig('Abbildungen/pressure_FS_preFFT.png')
##df_f
fish_sc_f = pd.DataFrame([fisher_score(df_f[i],df_f['Slugflow']) for i in col], columns=['0', '1'])
fish_sc_f['Messpunkt'] = col
fish_sc_f_melt = pd.melt(fish_sc_f, id_vars='Messpunkt', value_vars=['0', '1'], var_name='Slugging')
fs_f= sns.barplot(x='Messpunkt', y='value', hue='Slugging', data=fish_sc_f_melt)
fs_f.set_title("Fisher Score post FFT")
plt.savefig('Abbildungen/pressure_FS_postFFT.png')
#INFORMATION GAIN
##df_t
ig_t= skf.mutual_info_classif(df_t.iloc[:][col], df_t['Slugflow'])
plt.figure(num=n_f)
n_f = n_f+1
plt.bar(col, ig_t)
plt.xlabel("Sensor")
plt.ylabel("Information Gain [-]")
plt.title("Information Gain Drucksensoren preFFT")
##df_f
ig_f = skf.mutual_info_classif(df_f.iloc[:][col], df_f['Slugflow'])
plt.figure(num=n_f)
n_f = n_f+1
plt.bar(col, ig_f)
plt.xlabel("Sensor")
plt.ylabel("Information Gain [-]")
plt.title("Information Gain Drucksensoren postFFT")
#median
##df_t
df_s = df_t[df_t['Slugflow']==1]
df_n = df_t[df_t['Slugflow']==0]
df_median = df_t[col].median()
df_s_median = df_s[col].median()
df_n_median = df_n[col].median()
plt.figure(num=n_f)
n_f = n_f+1
plt.plot(df_median, "b--") #Median beide Klassen
plt.plot(df_s_median, "ro-")
plt.plot(df_n_median, "go-")
plt.xlabel("Sensor")
plt.ylabel("Druck [barg]")
plt.title("Mediane der Drucksensoren")
##df_f
df_s = df_f[df_f['Slugflow']==1]
df_n = df_f[df_f['Slugflow']==0]
df_median = df_f[col].median()
df_s_median = df_s[col].median()
df_n_median = df_n[col].median()
plt.figure(num=n_f)
n_f = n_f+1
plt.plot(df_median, "b--") #Median beide Klassen
plt.plot(df_s_median, "ro-")
plt.plot(df_n_median, "go-")
plt.xlabel("Sensor")
plt.ylabel("Druck [barg]")
plt.title("Mediane der Drucksensoren")


#Boxplot (https://seaborn.pydata.org/examples/horizontal_boxplot.html)
##df_t
plt.figure(num=n_f)
n_f = n_f+1
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x="value", y="Messpunkt", data=data_t, hue="Slugflow", whis=[0, 100], width=.6, palette="vlag")
##df_f
plt.figure(num=n_f)
n_f = n_f+1
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x="value", y="Messpunkt", data=data_f[data_f['Messpunkt']!='P4-B14'], hue="Slugflow", whis=[0, 100], width=.6, palette="vlag")
print("X")

#VIZ
# Initialize the figure



