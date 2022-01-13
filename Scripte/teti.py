import pandas as pd

path = 'Pressure/Raw/0912Test9_20_01.csv'
df = pd.read_csv(path, keep_default_na=False)
df.head(5)