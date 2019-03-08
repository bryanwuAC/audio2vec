import pickle
import torch
import hyperparameters as hps
from torch.autograd import Variable


def save_pickle(file_name, file_data):
    with open(file_name, "wb") as fp:
        pickle.dump(file_data, fp)


def load_pickle(file_name):
    with open(file_name, "rb") as fp:
        return pickle.load(fp)


def compute_reconstruction_loss(target, output, mask):
    return torch.mean(((target - output) * mask) ** 2)


def build_teacher_forcing_inputs(batch, z):
    sos = Variable(torch.stack([torch.zeros(batch.shape[2])] * hps.batch_size).cuda()).unsqueeze(0)
    batch = torch.cat([sos, batch], 0)

    max_len = batch.shape[0]
    z_stack = torch.stack([z] * max_len)
    teacher_forcing_inputs = torch.cat([batch, z_stack], 2)

    return teacher_forcing_inputs
