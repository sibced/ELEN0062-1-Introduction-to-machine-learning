"""
"""

import numpy as np
import torch

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .model import Model, Deeper

from . import COORDS_MEAN, COORDS_STD, TIME_START_MEAN, TIME_START_STD
from . import TESTSET, TRAINSET, POS_COLUMNS, VALIDATIONSET


class ProjectDS(Dataset):
    def __init__(self, dataset=None, transform=None) -> None:
        super().__init__()
        self.dataset = TRAINSET if dataset is None else dataset
        self.transform = transform

    def __getitem__(self, index: int):
        sample = {
            'coords': self.dataset.loc[index, POS_COLUMNS],
            'sender': self.dataset.loc[index, 'sender'] - 1,
            'time_start': self.dataset.loc[index, 'time_start'],
            'idx': index
        }

        if 'receiver' in self.dataset.columns:
            sample['receiver'] = self.dataset.loc[index, 'receiver'] - 1

        if self.transform:
            sample = self.transform(sample)

        # print('\n\n\n',sample,'\n\n\n')

        return sample

    def __len__(self) -> int:
        return len(self.dataset)


class Normalizer:
    def __init__(self, mean=None, std=None):
        self.mean = COORDS_MEAN if mean is None else mean
        self.std = COORDS_STD if std is None else std

    def __call__(self, sample):
        sender = sample['sender']

        # scale to [-1, 1]
        coords = sample['coords']
        coords[00:22] /= 5250.0
        coords[22:44] /= 3400.0

        # as tensor
        coords = torch.tensor(data=coords)

        # get second input
        sender_coords = torch.tensor([coords[sender], coords[sender + 22]])

        # get full input
        coords = torch.cat([coords, sender_coords])

        time_start = sample['time_start']
        time_start -= TIME_START_MEAN
        time_start /= TIME_START_STD

        # add time start
        coords = torch.cat([coords, torch.tensor([time_start])])

        # put in dataset
        sample['coords'] = coords

        return sample


class Augmenter:
    def __init__(self, prob=.5) -> None:
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() >= self.prob:
            return sample

        if np.random.rand() < 0.5:
            # flip vertically (symmetry)

            # flip all y values
            sample['coords'][22:44] *= -1.

            # flip sender y
            sample['coords'][45] *= -1.
        else:
            # flip horizontally & change teams
            coords = sample['coords']

            # flip all x values
            coords[00:22] *= -1.

            # flip sender x
            coords[44] *= -1.

            # flip teams x
            coords[00:11], coords[11:22] = coords[11:22], coords[00:11]

            # flip teams y
            coords[22:33], coords[33:44] = coords[33:44], coords[22:33]

            # put back into sample (<-useless)
            sample['coords'] = coords

            # flip sender idx
            sample['sender'] += 11
            sample['sender'] %= 22

            # flip receiver idx
            sample['receiver'] += 11
            sample['receiver'] %= 22

        return sample


def eval(dataldr: DataLoader, model: torch.nn.Module, criterion, device):
    """returns the averaged loss and the accuracy over a given dataset

    - dataldr: dataloader than give batches
    - model: model on device `device`
    - criterion: loss function object
    - device: a torch.device, either cpu or cuda:index
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        good = 0.
        total = 0.
        for batch in dataldr:
            inputs = batch['coords'].to(device)
            label = batch['receiver'].to(device)

            out = model(inputs)
            pred = out.argmax(dim=1)

            good += (pred == label).sum()
            total += inputs.shape[0]

            loss = criterion(out, label)
            total_loss += loss.item()
        return good / total, total_loss / len(dataldr.dataset)


def train(epoch_count, model, criterion, device, optimizer, train_ldr, test_ldr):
    """train a model over the train set and returns some performances metrics
    gathered during training.

    - model: model on device `device`
    - criterion: loss function object
    - device: a torch.device, either cpu or cuda:index
    - optimizer: an optimizer object
    - train_ldr: torch.data.utils.DataLoader used to perform training
    - test_ldr: torch.data.utils.DataLoader used to perform the evaluation

    returns `stats[i]` is [averaged train loss, averaged test loss, accuracy]
    after epoch i.
    """
    stats = []
    for i in range(epoch_count):
        model.train()
        train_loss = 0
        total = len(train_ldr.dataset)
        for batch_id, batch in enumerate(train_ldr):
            optimizer.zero_grad()

            inputs = batch['coords'].to(device)
            out = model(inputs)
            loss = criterion(out, batch['receiver'].to(device))

            print((f'Epoch: {i+1:2}/{epoch_count}, batch loss {batch_id+1:4}/{len(train_ldr)}'
                   f' loss:{loss.item()}'), end='\r')

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print((f'\nEpoch: {i+1:2}/{epoch_count} '
               f'average loss:{train_loss/total:.5E}'))

        acc_tst, loss_tst = eval(test_ldr, model, criterion, device)
        print(f'test set      : acc {acc_tst:4.3f}, loss: {loss_tst:.5E}')

        stats.append([train_loss/total, loss_tst, acc_tst])
    return stats


def main():
    train_ds = ProjectDS(TRAINSET, transforms.Compose(
        [Normalizer(), Augmenter(.5)]))
    train_ldr = DataLoader(train_ds, batch_size=128,
                           shuffle=True, num_workers=4)

    test_ds = ProjectDS(TESTSET, Normalizer())
    test_ldr = DataLoader(test_ds, batch_size=128,
                          shuffle=False, num_workers=4)

    valid_ds = ProjectDS(VALIDATIONSET, Normalizer())
    valid_ldr = DataLoader(valid_ds, batch_size=128,
                           shuffle=False, num_workers=4)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Deeper().to(device)
    # model.load_state_dict(torch.load('fname.pth'), strict=True)

    optim = torch.optim.Adadelta(model.parameters(), weight_decay=.2)
    lossfn = torch.nn.CrossEntropyLoss()

    stats = train(50, model, lossfn, device, optim, train_ldr, test_ldr)
    torch.save(torch.tensor(stats), 'fname_stats.pth')

    acc_trn, loss_trn = eval(train_ldr, model, lossfn, device)
    print(f'training set  : acc {acc_trn:4.3f}, loss: {loss_trn:.5E}')

    acc_tst, loss_tst = eval(test_ldr, model, lossfn, device)
    print(f'test set      : acc {acc_tst:4.3f}, loss: {loss_tst:.5E}')

    acc_val, loss_val = eval(valid_ldr, model, lossfn, device)
    print(f'validation set: acc {acc_val:4.3f}, loss: {loss_val:.5E}')

    torch.save(model.cpu().state_dict(), 'fname.pth')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('interrupted')
