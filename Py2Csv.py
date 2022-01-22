import pandas as pd
import numpy as np

Label_Data = np.load('Data\LabelTrain.npy')

Label_Data.tofile('Data\Train_Label_Data.csv',sep=',')

