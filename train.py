# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np

from networks import EncoderCNN, DecoderRNN, LigandVAE
from generators import queue_datagen
from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to input .smi file.")
parser.add_argument("-o", "--output_dir", required=True, help="Path to model save folder.")
args = vars(parser.parse_args())

cap_loss = 0.
caption_start = 4000
batch_size = 128

savedir = args["output_dir"]
os.makedirs(savedir, exist_ok=True)
smiles = np.load(args["input"])

import multiprocessing
multiproc = multiprocessing.Pool(6)
my_gen = queue_datagen(smiles, batch_size=batch_size, mp_pool=multiproc)
mg = GeneratorEnqueuer(my_gen, seed=0)
mg.start()
mt_gen = mg.get()

# Define the networks
encoder = EncoderCNN(5)
decoder = DecoderRNN(512, 1024, 29, 1)
vae_model = LigandVAE(use_cuda=True)

encoder.cuda()
decoder.cuda()
vae_model.cuda()

# Caption optimizer
criterion = nn.CrossEntropyLoss()
caption_params = list(decoder.parameters()) + list(encoder.parameters())
caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

encoder.train()
decoder.train()

# VAE optimizer
reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD


vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-4)
vae_model.train()

tq_gen = tqdm(enumerate(mt_gen))
log_file = open(os.path.join(savedir, "log.txt"), "w")
cap_loss = 0.
caption_start = 4000

for i, (mol_batch, caption, lengths) in tq_gen:

    in_data = Variable(mol_batch[:, :5])
    in_data = in_data.cuda()

    discrim_data = Variable(mol_batch[:, 5:].cuda())

    vae_optimizer.zero_grad()
    recon_batch, mu, logvar = vae_model(in_data, discrim_data)
    vae_loss = loss_function(recon_batch, in_data, mu, logvar)

    vae_loss.backward(retain_graph=True if i >= caption_start else False)
    p_loss = vae_loss.data[0]
    vae_optimizer.step()

    if i >= caption_start:  # Start by autoencoder optimization
        captions = Variable(caption.cuda())
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        decoder.zero_grad()
        encoder.zero_grad()
        features = encoder(recon_batch)
        outputs = decoder(features, captions, lengths)
        cap_loss = criterion(outputs, targets)
        cap_loss.backward()
        caption_optimizer.step()

    if (i + 1) % 5000 == 0:
        torch.save(decoder.state_dict(),
                   os.path.join(savedir,
                                'decoder-%d.pkl' % (i + 1)))
        torch.save(encoder.state_dict(),
                   os.path.join(savedir,
                                'encoder-%d.pkl' % (i + 1)))
        torch.save(vae_model.state_dict(),
                   os.path.join(savedir,
                                'vae-%d.pkl' % (i + 1)))

    if (i + 1) % 100 == 0:
        result = "Step: {}, caption_loss: {:.5f}, " \
                 "VAE_loss: {:.5f}".format(i + 1,
                                           float(cap_loss.data.cpu().numpy()) if type(cap_loss) != float else 0.,
                                           p_loss)
        log_file.write(result + "\n")
        log_file.flush()
        tq_gen.write(result)

    # Reduce the LR
    if (i + 1) % 60000 == 0:
        # Command = "Reducing learning rate".format(i+1, float(loss.data.cpu().numpy()))
        log_file.write("Reducing LR\n")
        tq_gen.write("Reducing LR")
        for param_group in caption_optimizer.param_groups:
            lr = param_group["lr"] / 2.
            param_group["lr"] = lr

    if i == 210000:
        # We are Done!
        log_file.close()
        break

# Cleanup
del tq_gen
mt_gen.close()
multiproc.close()
