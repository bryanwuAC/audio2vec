import torch
import hyperparameters as hps
import utils
import os
from dataloader import DataLoader
from models import Seq2seq

data_loader = DataLoader(hps.mfcc_path)
seq2seq = Seq2seq(data_loader.feature_dimension)
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=hps.lr)

for i in range(hps.num_epoch):
    batch, batch_mask = data_loader.get_batch(hps.batch_size)

    optimizer.zero_grad()
    generated_outputs = seq2seq(batch)
    loss = utils.compute_reconstruction_loss(batch, generated_outputs, batch_mask)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print("Epoch {}: loss is {}".format(i, loss))

    if i % 100 == 0:
        os.makedirs(hps.model_dir, exist_ok=True)
        torch.save(seq2seq, "{}/sec2sec_{}.pkl".format(hps.model_dir, i))
