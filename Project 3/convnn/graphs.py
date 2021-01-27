import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train = pd.read_csv('convnn/second_train.csv')

fig, ax1 = plt.subplots()

x = np.arange(len(train))+15

ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracies')
ax1.plot(x, train[' train accuracy'], '-')
ax1.plot(x, train[' test accuracy'], '-.')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('loss')  # we already handled the x-label with ax1
ax2.plot(x, train[' train loss'], '-', label='training')
ax2.plot(x, train[' test loss'], '-.', label='testing')

plt.title(f'Learning metrics over the epochs 15 to {x.max()}')
plt.legend(loc='upper center')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()