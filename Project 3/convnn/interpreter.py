""

import torch

from torch.nn.functional import conv2d


class Interpreter:

    def __init__(self, std=.7):
        ""
        # auto adapt size
        kernel = 1
        edge_val = 1.

        scale = 2 * std**2

        # Gaussian filter not normalized
        while edge_val > .2:
            value_x = (torch.arange(kernel) - (kernel - 1) / 2).repeat(kernel, 1)
            value_y = value_x.T
            value = -value_x**2 - value_y**2
            self.weights = torch.exp(value / scale).view(1, 1, kernel, kernel)
            self.pad = kernel // 2

            edge_val = self.weights[0, 0, 0, kernel // 2]
            kernel += 2

    def to(self, *args, **kwargs):
        self.weights = self.weights.to(*args, **kwargs)
        return self

    def filter(self, img):
        return conv2d(img, self.weights, padding=self.pad)

    def interpret(self, sample):
        img = sample['output']
        img = self.filter(img)

        # probs[b, i] = prob of player (i+1) receiving the ball at image b
        probs = torch.zeros((0, 22)).to(img.device)
        # ranks[i] = amout of time the i-th rank was selected
        batch_rank = torch.zeros((22,), dtype=int).to(img.device)

        for b in range(img.shape[0]):
            pos = sample['pos'][b]

            # computes the probability for each player
            prob = img[b, 0, pos[:, 0], pos[:, 1]].view(1, -1)
            probs = torch.cat([probs, prob])

            # indices[i] = position in prob of the (i+1)-th biggest value
            indices = torch.argsort(prob, descending=True).view(-1)

            if 'receiver_id' in sample:
                # rank is the position of receiver_id in indices
                # it is zero is the receiver has the max probability
                rank = (indices == sample['receiver_id'][b]).nonzero()
                batch_rank[rank] += 1

        return probs, batch_rank

if __name__ == "__main__":
    from .model import Model
    from .dataloader import CSVDataset, ToImage
    from torch.utils.data import DataLoader
    from torchvision import transforms

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model()
    model.load_state_dict(torch.load('small3.pth'), strict=True)
    model = model.to(device)

    dataloader = DataLoader(CSVDataset.validation_set(transforms.Compose([ToImage()])),
                            batch_size=128, shuffle=False)
    
    model.eval()

    # evaluate the effect of std on the classifier
    for s in torch.arange(.2, 2.4, .2):
        interpreter = Interpreter(std=s).to(device)
        ranking = torch.zeros((22,), dtype=int).to(device)
        with torch.no_grad():
            for data in dataloader:
                image = data['image']            
                data['output'] =  model(image.to(device))
                _, rank = interpreter.interpret(data)
                ranking += rank
        total = ranking.sum()
        if total != len(dataloader):
            print(total, len(dataloader))
        print(f's={s:4.2}, ranks: {ranking[:5].float() / total}')