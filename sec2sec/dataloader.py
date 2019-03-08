import numpy as np
import utils
import torch
from torch.autograd import Variable


class DataLoader:
    def __init__(self, data_path):
        raw_data = utils.load_pickle(data_path)
        self.feature_dimension = raw_data[0].shape[0]
        self.max_len = self.get_max_len(raw_data)

        # normalized_data = self.normalize(raw_data)
        normalized_data = [data.T for data in raw_data]
        self.data, self.mask = self.apply_padding(normalized_data)

    def get_max_len(self, raw_data):
        lengths = [audio.shape[1] for audio in raw_data]
        return max(lengths)

    def normalize(self, data):
        for i in range(len(data)):
            mean = np.mean(data[i], axis=1)
            std = np.std(data[i], axis=1)
            # Convert each frame from column to row
            data[i] = (data[i].T - mean) / std
        return data

    def apply_padding(self, data):
        padded_data = []
        padding_mask = []
        for i in range(len(data)):
            new_data = np.ones((self.max_len, self.feature_dimension))
            new_mask = np.zeros((self.max_len, self.feature_dimension))

            new_data[:data[i].shape[0]] = data[i]
            new_mask[:data[i].shape[0]].fill(1)

            padded_data.append(new_data)
            padding_mask.append(new_mask)
        return np.stack(padded_data, axis=1), np.stack(padding_mask, axis=1)

    def get_batch(self, batch_size):
        indices = np.random.choice(self.data.shape[1], batch_size)
        batch = Variable(torch.from_numpy(self.data[:, indices, :]).cuda().float())
        batch_mask = torch.from_numpy(self.mask[:, indices, :]).cuda().float()
        return batch, batch_mask
