import dvc.api
import numpy as np

Features = modelpkl = dvc.api.read(
    'Data/FeatureDataTest.npy',
    repo='https://github.com/marlusmate/B_SimOpt.git',
    mode='rb')

Labels = modelpkl = dvc.api.read(
    'Data/LabelTest.npy',
    repo='https://github.com/marlusmate/B_SimOpt.git',
    mode='rb')

np.save('FeatureDataTest.npy', Features)
np.save('LabelTest.npy', Labels)
