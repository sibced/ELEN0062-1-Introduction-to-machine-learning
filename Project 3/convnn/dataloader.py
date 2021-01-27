import random

import torch
import pandas as pd

from os.path import abspath, dirname, join
from typing import Any, Dict

from torch.utils.data import Dataset


# dir where the csv files are
ROOT = dirname(dirname(abspath(__file__)))
DATA_DIR = join(ROOT, 'dataset')


class CSVDataset(Dataset):
    "gives a dataset representation from a csv file"

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.transform = transform

        self.data = dataframe

        self.abscissa = self.data[[f'x_{i}' for i in range(1, 23)]]
        self.ordinate = self.data[[f'y_{i}' for i in range(1, 23)]]
        self.sender = self.data['sender']
        if 'receiver' in self.data:
            self.receiver = self.data['receiver']
        else:
            self.receiver = pd.Series(dtype=int)  # empty data
        self.time_start = self.data['time_start']

    @staticmethod
    def training_set(transform):
        dataframe = pd.read_csv(join(DATA_DIR, 'training.csv'))
        return CSVDataset(dataframe, transform)

    @staticmethod
    def test_set(transform):
        dataframe = pd.read_csv(join(DATA_DIR, 'testing.csv'))
        return CSVDataset(dataframe, transform)

    @staticmethod
    def validation_set(transform):
        dataframe = pd.read_csv(join(DATA_DIR, 'validation.csv'))
        return CSVDataset(dataframe, transform)

    @staticmethod
    def task_set(transform):
        dataframe = pd.read_csv(join(ROOT, 'input_test_set.csv'), index_col=0)
        return CSVDataset(dataframe, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pos = torch.tensor([[y, x] for x, y in zip(self.abscissa.iloc[idx], self.ordinate.iloc[idx])])

        sample = {
            'sender_id': self.sender[idx] - 1,  # get id in [0, 22[
            'idx': idx,
            'pos': pos
        }

        if not self.receiver.empty:
            sample['receiver_id'] = self.receiver[idx] - 1  # get id in [0, 22[

        if self.transform:
            sample = self.transform(sample)


        return sample


class ToImage:
    # def __init__(self, size=(3, 256, 256), bounds=(3400, 5350)) -> None:
    def __init__(self, size=(3, 64, 64), bounds=(4100, 5600)) -> None:
        """
        - size: (channel, height, width)
        - bounds: (height, width)
        """
        self.size = size
        self.bounds = bounds

    def __call__(self, sample: Dict[str, Any]) -> Any:
        """
        """

        pos = sample['pos']

        # map to grid indices
        pos[:, 0] = (pos[:, 0] + self.bounds[0]) * self.size[1] / self.bounds[0] // 2
        pos[:, 1] = (pos[:, 1] + self.bounds[1]) * self.size[2] / self.bounds[1] // 2
        pos = pos.to(int)

        sample['pos'] = pos


        img = torch.zeros(self.size)

        team1 = pos[:11, :]
        img[0, team1[:, 0], team1[:, 1]] = 1.

        sender = pos[sample['sender_id']]
        img[1, sender[0], sender[1]] = 1.

        team2 = pos[11:, :]
        img[2, team2[:, 0], team2[:, 1]] = 1.

        sample['image'] = img


        if 'receiver_id' in sample:
            label = torch.zeros(1, *self.size[1:])
            receiver = pos[sample['receiver_id']]
            label[0, receiver[0], receiver[1]] = 1.
            sample['label'] = label

        return sample


class Augmenter(object):
    """
    Augmenter: augment the dataset
    """

    def __init__(self, proba) -> None:
        self.proba = proba

    def __call__(self, sample: Dict[str, Any]) -> Any:
        if random.random() >= self.proba:
            return sample

        # todo: reverse x and team ?

        sample['image'] = sample['image'].flip(1)
        sample['label'] = sample['label'].flip(1)

        _, h, _ = sample['image'].shape

        # flip y
        sample['pos'][:, 0] = h - 1 - sample['pos'][:, 0]

        return sample
