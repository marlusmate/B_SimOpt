import dvc.api

Features_url = modelpkl = dvc.api.read(
    'FeatureDataTest.npy',
    repo='https://github.com/marlusmate/B_SimOpt.git',
    mode='rb')

