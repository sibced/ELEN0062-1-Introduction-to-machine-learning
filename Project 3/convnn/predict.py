import time

import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from .dataloader import ToImage, CSVDataset
from .model import Model
from .interpreter import Interpreter


def write_submission(predictions=None, probas=None, estimated_score=0, file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    predictions: array [n_predictions, 1]
        `predictions[i]` is the prediction for player 
        receiving pass `i` (or indexes[i] if given).
    probas: array [n_predictions, 22]
        `probas[i,j]` is the probability that player `j` receives
        the ball with pass `i`.
    estimated_score: float [1]
        The estimated accuracy of predictions. 
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """   

    if date: 
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    if predictions is None and probas is None:
        raise ValueError('Predictions and/or probas should be provided.')

    n_samples = 3000
    if indexes is None:
        indexes = np.arange(n_samples)

    if probas is None:
        print('Deriving probabilities from predictions.')
        probas = np.zeros((n_samples,22))
        for i in range(n_samples):
            probas[i, predictions[i]-1] = 1

    if predictions is None:
        print('Deriving predictions from probabilities')
        predictions = np.zeros((n_samples, ))
        for i in range(n_samples):
            mask = probas[i] == np.max(probas[i])
            selected_players = np.arange(1,23)[mask]
            predictions[i] = int(selected_players[0])


    # Writing into the file
    with open(file_name, 'w') as handle:
        # Creating header
        header = '"Id","Predicted",'
        for j in range(1,23):
            header = header + '"P_{:0.0f}",'.format(j)
        handle.write(header[:-1]+"\n")

        # Adding your estimated score
        first_line = '"Estimation",{},'.format(estimated_score)
        for j in range(1,23):
            first_line = first_line + '0,'
        handle.write(first_line[:-1]+"\n")

        # Adding your predictions
        for i in range(n_samples):
            line = "{},{:0.0f},".format(indexes[i], predictions[i])
            pj = probas[i, :]
            for j in range(22):
                line = line + '{},'.format(pj[j])
            handle.write(line[:-1]+"\n")

    return file_name

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

model_path = 'final_cnn.pth'

model = Model()
if model_path is not None:
    model.load_state_dict(torch.load(model_path), strict=True)
model = model.to(device)

taskset = CSVDataset.task_set(transforms.Compose([ToImage()]))
taskldr = DataLoader(taskset, batch_size=256, shuffle=False)

interpreter = Interpreter(std=.5).to(device)

model.eval()
with torch.no_grad():
    prob_total = torch.zeros((0, 22)).to(device)
    for data in taskldr:
        out = model(data['image'].to(device))
        data['output'] = out
        prob, _ = interpreter.interpret(data)
        # normalize
        prob /= prob.sum(dim=1).view(-1, 1)

        # add to total
        prob_total = torch.cat([prob_total, prob])
    
    fname = write_submission(
        probas=prob_total.cpu().numpy(),
        estimated_score=.4,
        file_name="cnn")
    print('Submission file "{}" successfully written'.format(fname))




