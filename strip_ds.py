"""
results:
resulting len: 7576

for test set (3000):
    len(strip(df, 311, 501)) = 3000

for training set (8682):
    len(strip(df, 281, 601)) = 8682

-> original semi dims: 5250, 3400
-> min margins: (311, 601)
-> bounds: (5600, 4100)
"""

import sys

import pandas as pd

def strip(df, dx=0, dy=None):
    if dy is None:
        dy = dx
    for i in range(1, 23):
        df = df.loc[df[f'x_{i}'] > -5250-dx]
        df = df.loc[df[f'x_{i}'] <  5250+dx]
        df = df.loc[df[f'y_{i}'] > -3400-dy]
        df = df.loc[df[f'y_{i}'] <  3400+dy]
    return df

def clip(df):
    for i in range(1, 23):
        df[f'x_{i}'].clip(lower=-5250, upper=5249, inplace=True)
        df[f'y_{i}'].clip(lower=-3400, upper=3399, inplace=True)

# save to file
if __name__ == '__mian__' and len(sys.argv) > 1:
    # load raw file
    inputs = pd.read_csv('input_training_set.csv', index_col=-1, header=0)
    labels = pd.read_csv('output_training_set.csv', header=0)
    # fuse inputs and outputs
    inputs = inputs.join(labels)

    print(f'raw size: {len(inputs)}')

    dx, dy = 1, 1
    if len(sys.argv) > 2:
        dx = int(sys.argv[2])
        dy = dx
    if len(sys.argv) > 3:
        dy = int(sys.argv[3])

    stripped = strip(inputs, dx, dy)
    print(f'stripped size: {len(stripped)}')
    stripped.to_csv(sys.argv[1], index=False)
