"""
"""

from sys import argv as args
import pandas as pd
from matplotlib import pyplot as plt

if __name__ != "__main__":
    print(__doc__)
    raise RuntimeError('stats should not be imported')

if len(args) < 2:
    print(__doc__)
    raise RuntimeError('need at leat one argument')

inputs = pd.read_csv(args[1])

has_receiver = 'receiver' in inputs.columns

cols_x = [f'x_{i}' for i in range(1, 23)]
cols_y = [f'y_{i}' for i in range(1, 23)]
cols = cols_x + cols_y

mean = inputs[cols].mean()
std = inputs[cols].std()

print('position mean')
print(mean)
print('position standard deviation (unbiased)')
print(std)

sender_rep = inputs[['sender']].value_counts().sort_index()
print(sender_rep)

if has_receiver:
    receiver_rep = inputs[['receiver']].value_counts().sort_index()
    print(receiver_rep)

# todo separate x and y
plt.plot(inputs[cols_x[:11]].mean(), '-o')
plt.plot(inputs[cols_x[:11]].std(),'-o')
# plt.plot(inputs[cols_y[:11]].mean(), '-o')
# plt.plot(inputs[cols_y[:11]].std(),'-o')
plt.legend(['mean (1)', 'std (1)'])
plt.show()

# plt.plot(inputs[cols_x[11:]].mean(), '-o')
# plt.plot(inputs[cols_y[11:]].std(),'-o')
# plt.legend(['mean (2)', 'std (2)'])
# plt.show()
