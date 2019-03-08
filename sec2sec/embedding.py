import torch
import hyperparameters as hps
import utils
import numpy as np
from dataloader import DataLoader


def embedding(model_path, input_data_path):
    sec2sec = torch.load(model_path)
    dataloader = DataLoader(input_data_path)
    audios = torch.from_numpy(dataloader.data).cuda().float()
    num_audios = audios.shape[1]

    encoder = sec2sec.encoder

    embedding_vectors = np.zeros((num_audios, hps.latent_vector_length))
    batch_start_index = 0
    while batch_start_index + hps.batch_size < num_audios:
        vector = encoder(audios[:, batch_start_index:batch_start_index + hps.batch_size, :], hps.batch_size).detach()
        embedding_vectors[batch_start_index:batch_start_index + hps.batch_size, :] = vector.cpu().numpy()
        batch_start_index += hps.batch_size
        print("Embedded {} audios.".format(batch_start_index))

    vector = encoder(audios[:, batch_start_index:num_audios, :], num_audios - batch_start_index).detach()
    embedding_vectors[batch_start_index:, :] = vector.cpu().numpy()
    print("Embedded {} audios.".format(num_audios))
    return embedding_vectors

if __name__ == "__main__":
    model_path = "{}/sec2sec_mfcc_{}.pkl".format(hps.model_dir, 5900)
    input_data_path = hps.mfcc_path
    output_data_path = "mfcc_embedding_vector.pkl"
    embedding_vectors = embedding(model_path, input_data_path)
    utils.save_pickle(output_data_path, embedding_vectors)
