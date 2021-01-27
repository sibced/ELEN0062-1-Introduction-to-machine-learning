from os.path import dirname, abspath, join

import pandas as pd

# dir where the csv files are
ROOT = dirname(dirname(abspath(__file__)))
DATA_DIR = join(ROOT, 'dataset')

TRAINSET = pd.read_csv(join(DATA_DIR, 'training.csv'))

TESTSET = pd.read_csv(join(DATA_DIR, 'testing.csv'))

VALIDATIONSET = pd.read_csv(join(DATA_DIR, 'validation.csv'))

X_COLUMNS = [f'x_{i}' for i in range(1, 23)]
Y_COLUMNS = [f'y_{i}' for i in range(1, 23)]
POS_COLUMNS = X_COLUMNS + Y_COLUMNS

COORDS_MEAN = TRAINSET[POS_COLUMNS].mean()
COORDS_STD = TRAINSET[POS_COLUMNS].std()

TIME_START_MEAN = TRAINSET['time_start'].mean()
TIME_START_STD = TRAINSET['time_start'].std()
